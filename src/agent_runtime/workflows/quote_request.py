"""Route a quote agent's decisions through a conversation with the human.

The quote interpreter agent interprets the request and answers; the graph routes
its decision and pauses safely. No business-field rules decide whether clarification is needed.
The caller owns checkpoint persistence. This local teaching workflow never
submits a quote to a supplier.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from agent_runtime.agents.quote_interpreter import build_agent, input_context
from agent_runtime.harness.model_factory import build_model


class QuoteState(TypedDict, total=False):
    """Retain the original request and human context across resumable pauses.

    The fields are intentionally free-form where the model owns interpretation:
    this graph does not impose a fixed name, email, quantity, or service schema.
    ``total=False`` matches LangGraph's incremental node updates; ``assess``
    initializes the decision and status before downstream nodes consume them.
    """

    # The user's initial request, preserved as supplied so assessment can revisit
    # all original context after each clarification.
    messages: Annotated[list[BaseMessage], add_messages]
    values: dict
    # Ordered question/answer pairs. Answers remain unaltered so the model can
    # interpret the human's wording rather than a normalized surrogate.
    conversation: list[dict]
    # Latest validated QuoteDecision, checkpointed before ask may interrupt.
    decision: dict
    # ``collecting``, ``completed``, or ``cancelled`` lifecycle state.
    status: str
    # Local summary only; this workflow never submits it to an external supplier.
    request: dict | None


def build_workflow(model=None, *, checkpointer=None):
    """Prepare a workflow that helps a human clarify a quote request.

    The application calls this factory before starting a conversation. It gets
    a runnable workflow that can ask for missing context, reconsider each
    answer, and finish with a prepared scope summary or explicit cancellation.
    Constructing the workflow does not begin the conversation.

    The separate ask node prevents resume from repeating a provider call: the
    validated decision is stored at the assess boundary before the interrupt.
    All answers return to the model, so no predetermined question sequence
    governs live behavior. Invalid model output fails visibly instead of
    completing. Direct callers supply a checkpointer for pause/resume;
    conversation entry points attach their retained saver before execution.
    This factory never allocates persistence; durability depends on the caller.

    ``model`` is a chat model supporting bind_tools/invoke, either a live
    provider or the scripted fixture. Return a compiled LangGraph; building it
    does not contact the provider. Invoke it with ``values`` and a config with
    configurable.thread_id; resume the same graph/thread with
    Command(resume=answer) so it can find the paused conversation.
    """
    # Model selection belongs to the factory; the workflow supplies its role.
    if model is None:
        model = build_model(caller="workflow")
    # Build the reusable model wrapper; assess below triggers the actual call.
    agent = build_agent({"model": model})

    def begin(state):
        """Normalize a new client request once, before any clarification pause."""
        # CLI JSON fields and a free-form browser request share this boundary.
        # Resume starts at ask, so it never replaces the original request.
        values = state.get("values", {})
        if state.get("messages"):
            text = state["messages"][-1].content
            try:
                decoded = json.loads(text)
            except (ValueError, TypeError):
                decoded = None
            values = decoded if isinstance(decoded, dict) else {"description": text}
        return {"values": values, "conversation": [], "request": None}

    def assess(state, config):
        """Determine whether the request needs clarification or is ready to finish.

        The graph calls this node for the initial request and after each human
        answer. It should produce one validated decision: a question that helps
        resolve uncertainty, or a summary ready for the completion node. Each
        answer needs reassessment because it may leave an issue unresolved or
        introduce a new contradiction.

        The harness retains the complete initial request and ordered answers.
        The agent owns their model-facing representation and validates its own
        output protocol before returning a decision. Malformed model responses
        raise from the agent and cannot select an edge or silently complete.

        ``state`` is QuoteState; ``config`` carries execution settings/callbacks.
        Return a partial update so the validated decision is checkpointed before
        ``ask`` reaches its interrupt. This prevents a resume from repeating the
        preceding assessment call.
        """
        # Supply task data, not a HumanMessage template or a model tool schema.
        # The workflow controls retained context while the agent encapsulates how
        # its chosen model understands that context and expresses a decision.
        decision = agent.invoke(
            {
                "initial_request": state["values"],
                "conversation": state.get("conversation", []),
            },
            config=config,
        )
        # Save a plain dictionary for checkpointing. Clear any final summary
        # until the complete node explicitly produces one.
        return {
            "decision": decision.model_dump(),
            "status": "collecting",
            "request": None,
        }

    def ask(state):
        """Obtain the human's input needed to resolve the model's uncertainty.

        The graph calls this node when assess chooses clarification. Its goal
        is to give the human a chance to answer or abandon the request, then
        retain an ordinary answer for reassessment. Receiving an answer alone
        does not establish that the request is ready to complete.

        This node contains no provider call.  Consequently, resuming an
        interrupt uses the already checkpointed decision and records one answer;
        the next assessment happens only after the answer is returned. The
        cancellation sentinel clears the current in-memory draft and
        conversation, while earlier checkpoints and execution traces remain
        available to the checkpointer's history.

        ``state`` contains the validated decision saved by assess. Return only
        changed fields: extended conversation for an ordinary answer, or cleared
        draft fields and cancelled status for the exact /cancel sentinel.
        """
        decision = state["decision"]
        # First pass: interrupt raises LangGraph's GraphInterrupt internally.
        # It does NOT return here: answer is not assigned and the code below
        # does not run. LangGraph handles the exception and returns the question
        # payload to the graph caller, where the client obtains a human response.
        #
        # Resume: the caller invokes this same graph/thread with
        # Command(resume=human_response). LangGraph runs ask FROM THE BEGINNING,
        # including the decision assignment above. When execution reaches this
        # interrupt again, it returns the saved human_response instead of
        # raising. Only then is answer assigned and the code below executes.
        #
        # This is replay, not Python yield or restoration of a suspended stack.
        # Code before interrupt runs again, so avoid side effects there. The
        # completed assess node is not replayed merely to resume this ask node.
        answer = interrupt(
            {
                "kind": "question",
                "question": decision["text"],
                "reason": decision["reason"],
                "cancel": "/cancel",
            }
        )
        if answer == "/cancel":
            # Current state is cleared; earlier checkpoints/traces still exist.
            return {
                "values": {},
                "conversation": [],
                "decision": {},
                "request": None,
                "status": "cancelled",
                "messages": [AIMessage(content="Quote clarification cancelled.")],
            }
        # There is no list-append reducer on conversation: a returned list
        # replaces it. Include all earlier pairs so the next assessment can
        # reconsider unresolved issues instead of seeing only the latest answer.
        return {
            "conversation": [
                *state.get("conversation", []),
                {
                    "question": decision["text"],
                    "answer": answer,
                },
            ]
        }

    def complete(state):
        """Finish clarification and provide the prepared quote request to the caller.

        The graph calls this node when assess reports that the request is clear
        enough and no further human clarification is needed. Its purpose is to
        turn that decision into a finished workflow result: the caller can see
        that collection is complete and read the prepared scope summary.

        ``state`` contains the validated model decision. This node returns a
        partial update setting status to completed and copying the decision's
        text into request.summary. LangGraph merges the update into state,
        preserving the original values and conversation, then follows the edge
        from complete to END.

        Completion means request preparation is finished. It does not calculate
        a price, check feasibility, or submit the request to a supplier.
        """
        # This is model-authored scope, not a supplier-issued quote.
        return {
            "status": "completed",
            "request": {"summary": state["decision"]["text"]},
            "messages": [AIMessage(content=state["decision"]["text"])],
        }

    def route_after_assessment(state):
        """Direct the workflow to the next step required by the assessment.

        LangGraph calls this after assess finishes so a request needing human
        input reaches ask, while a request the model considers ready reaches
        complete. The purpose is to act on the saved decision without making
        a second assessment that could disagree with it.

        ``state`` is graph state after assess's update has been merged into it.
        Return a node name, not a state update. This callback does not call the
        model; QuoteDecision validation has already rejected other actions.
        """
        return state["decision"]["action"]

    def route_after_answer(state):
        """Honor cancellation or ensure the human's answer gets reassessed.

        LangGraph calls this after ask returns. It should stop an abandoned
        conversation and send every ordinary answer back for model judgment,
        rather than treating the mere receipt of an answer as completion.

        ``state`` includes ask's merged update. Return LangGraph's END sentinel
        for cancelled status, otherwise the assess node name. Ordinary answers
        leave status as collecting and always trigger a fresh assessment.
        This callback runs after ask finishes, not while its interrupt is paused.
        """
        # Cancellation clears decision, so route on status rather than reading
        # an action from the discarded decision dictionary.
        if state["status"] == "cancelled":
            return END
        return "assess"

    # The graph makes model interpretation, human interruption, and finalization
    # separate states. In particular, assess -> ask stores the decision before
    # pausing, while ask -> assess is the only path that reconsiders an answer.
    # Possible flows (labels are routing conditions; the left arrow loops back):
    #
    # START -> assess -- action=complete --> complete -> END
    #            ^  |
    #            |  | action=ask
    #            |  v
    #            |  ask -- answer=/cancel ------------> END
    #            |  |
    #            |  | any other answer
    #            +--+
    #
    # ask pauses for the human and resumes with their answer before routing.
    # Each normal answer triggers a fresh assessment, which may ask again.
    # Invalid model decisions raise in assess; they are not completion paths.
    #
    graph = StateGraph(QuoteState)
    graph.add_node("assess", assess)
    graph.add_node("ask", ask)
    graph.add_node("complete", complete)
    graph.add_node("begin", begin)
    graph.add_edge(START, "begin")
    graph.add_edge("begin", "assess")
    # The agent contract restricts this edge to ask or complete; any model
    # protocol failure propagates through assess before routing is attempted.
    graph.add_conditional_edges("assess", route_after_assessment, ["ask", "complete"])
    # Cancellation is terminal and clears the current state. A normal answer
    # returns to assess so the model can reconsider all unresolved issues.
    graph.add_conditional_edges("ask", route_after_answer, ["assess", END])
    graph.add_edge("complete", END)
    # Bind only supplied persistence; conversation entry points can attach it later.
    compiled = graph.compile(checkpointer=checkpointer)

    def preview_context(state):
        """Project the next assessment's known input, excluding the human answer."""
        if state.get("status") != "collecting":
            return input_context()
        # The pending question is known; its answer is the next user input and
        # deliberately absent. The actual resume inserts it into this same pair.
        conversation = [*state.get("conversation", [])]
        if state.get("decision", {}).get("text"):
            conversation.append({"question": state["decision"]["text"]})
        return input_context(
            {"initial_request": state.get("values", {}), "conversation": conversation}
        )

    compiled.preview_context = preview_context
    return compiled
