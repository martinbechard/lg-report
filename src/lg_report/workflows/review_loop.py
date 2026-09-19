"""Enforce author → judge → revise/finish using an explicit LangGraph StateGraph.

One LLM is shared; author and judge have separate message histories. The graph,
not the LLM, enforces the revision limit and decides whether another round runs.
Only the final answer enters the user's conversation. Intermediate model calls
remain visible in the execution trace. See samples/review_loop/README.md.
Read this module in execution order: begin → write → assess → route. A rejection
loops back to write; approval or exhaustion goes to finish. For example, a run
can take author(1) → judge(revise) → author(2) → judge(approve) → finish.
Building the graph only registers these functions; invoking it executes them.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from typing import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langgraph.graph import END, START, StateGraph

from lg_report.agents import evidence_judge, review_author


class ReviewState(TypedDict, total=False):
    """Keep user conversation separate from internal draft and review histories.

    List fields use replacement, not append reducers: each node returns the full
    updated role history. This prevents duplicated drafts on a revision cycle.

    total=False allows the initial input to contain just messages. It does not
    supply defaults or validate values at runtime: begin initializes the other
    fields before any downstream node reads them. Nodes return partial updates;
    LangGraph retains state keys that a node does not include in its return value.
    """

    # The external chat: previous user/final-answer exchanges plus the new request.
    # Internal drafts and judge comments stay out of this list until finalization.
    messages: list[BaseMessage]
    # The author's working conversation includes its drafts and received feedback.
    author_history: list[BaseMessage]
    # The judge sees evidence, candidate drafts, and its own previous evaluations.
    # It never receives the author's private system instruction.
    judge_history: list[BaseMessage]
    # Number of completed drafts, not total LLM calls (each draft also gets judged).
    round: int
    # Latest candidate text, kept separate so the judge need not find it in history.
    draft: str
    # Validated Review serialized as a dict: verdict, rationale, and feedback.
    review: dict
    # pending until finish; approved or limit_reached describes content review.
    # A successfully executed graph can still return an unapproved draft.
    outcome: str


def build_workflow(model, *, max_rounds: int = 3, first_draft_high_level: bool = False):
    """Give the application an answer that has passed through draft review.

    Compile an author/judge cycle that can revise a draft up to the configured
    budget and returns an explicit approval or limit-reached outcome.

    max_rounds counts drafts, including the first; at least one is required.
    first_draft_high_level is an explicit teaching option. It asks the author for
    an overview on round one but never forces a judge verdict. The judge can
    approve immediately if an overview genuinely satisfies the user's request.
    model is a single shared chat-model object. Agent histories live in state,
    not inside the provider adapter. No second LLM is created for the judge.
    Return an executable compiled graph; construction performs no provider calls.
    Invalid judge JSON fails visibly instead of selecting an arbitrary edge.
    There is no checkpointer here: this lesson loops within one invocation and
    does not implement durable pause/resume. Provider/validation errors propagate.
    """
    # Python treats bool as an int, but True is not a meaningful draft budget.
    # Reject booleans, non-integers, and zero/negative limits before compiling;
    # otherwise the workflow could start with no well-defined review allowance.
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds < 1
    ):
        raise ValueError("max_rounds must be a positive integer")
    # These are lightweight role runnables around the SAME model. Keeping them
    # in separate files makes their instructions inspectable and gives tracing
    # distinct author/judge operations without creating separate model clients.
    author = review_author.build_agent(model)
    judge = evidence_judge.build_agent(model)

    def begin(state: ReviewState):
        """Start a fresh review cycle for the caller's latest conversation turn.

        ``state["messages"]`` is the external conversation supplied at invocation.
        Return a partial state update that preserves that conversation and resets
        internal histories/counters before LangGraph schedules the author.

        convert_to_messages accepts the dictionaries/tuples used by clients and
        yields message objects; later nodes can safely read .type and .content.
        Returning new lists avoids mutating the caller's conversation in place.
        """
        # Every user turn starts a new review cycle. Conversation supplies prior
        # user/final-answer history, but earlier internal reviews must not leak in.
        return {
            "messages": convert_to_messages(state["messages"]),
            "author_history": [],
            "judge_history": [],
            "round": 0,
            "draft": "",
            "review": {},
            "outcome": "pending",
        }

    def write(state: ReviewState):
        """Prepare a candidate answer, or correct it after the judge requests changes.

        LangGraph calls this node first after begin and again on a revision edge.
        The candidate goes to the judge next; it is not yet the user's answer.

        Read user messages on round zero. On later rounds, reuse the author-only
        history and append the judge's validated feedback. Return the updated
        author history, latest draft, and completed-draft count as state updates.
        """
        # Zero means no draft exists yet, so there is no useful feedback to supply.
        # Any positive round means the conditional edge has requested a revision.
        if state["round"] == 0:
            instruction = "Write an answer to the conversation's latest request."
            # The sample deliberately starts shallow to demonstrate useful feedback;
            # this does not ask for false facts or force a predetermined rejection.
            if first_draft_high_level:
                instruction += " For this first draft, give only a high-level overview; leave the detailed drill-down for revision."
            # LangChain's HumanMessage is a user-role message container, not an
            # interactive prompt or a human callback. Here application-generated
            # instructions enter the author's model context through that role.
            history = [*state["messages"], HumanMessage(content=instruction)]
        else:
            # Previous draft and original request remain in the author history.
            # Feed back the judge's actual decision, not an application-authored fix.
            history = [
                *state["author_history"],
                HumanMessage(
                    content="Revise the draft to address this review. Retain supported material.\n"
                    + json.dumps(state["review"])
                ),
            ]
        # Invoke is the actual paid model-call boundary in live mode. Nested
        # runnable callbacks reach the recorder, so every revision is accounted.
        response = author.invoke(history)
        # Store both the full AIMessage (history/usage) and visible text (review).
        # Increment only after the call succeeds; exceptions do not count as drafts.
        return {
            "author_history": [*history, response],
            "draft": response.text,
            "round": state["round"] + 1,
        }

    def assess(state: ReviewState):
        """Decide whether the candidate meets the user's request or needs revision.

        After each author turn, ``state`` provides the draft, original conversation,
        and prior judge history. Return updated judge history and a parsed review
        for route to choose the next node. This model assessment is not an
        independent factual guarantee; parsing only checks the decision contract.

        Repeating the original request/evidence keeps the rubric anchored to the
        user's needs instead of letting earlier feedback become a different task.
        Prior judge exchanges remain visible, including whether fixes were made;
        the system prompt explicitly asks it to evaluate the CURRENT candidate.
        """
        # Serialize role/content pairs as evidence inside one human message, not
        # as new system instructions. Attachments are already part of that content.
        # This preserves the entire conversation; it does not fetch outside facts.
        request = json.dumps(
            [
                {"role": message.type, "content": message.content}
                for message in state["messages"]
            ]
        )
        history = [
            *state["judge_history"],
            HumanMessage(
                content=f"Original conversation and evidence:\n{request}\n\nCURRENT draft:\n{state['draft']}"
            ),
        ]
        # This role runnable calls the chat model and returns its AIMessage.
        # response.text is model-authored JSON text, not an executed tool result.
        response = judge.invoke(history)
        # Strict validation protects the conditional edge from misspelled verdicts,
        # missing feedback, or an apparent approval embedded in arbitrary prose.
        review = evidence_judge.Review.model_validate_json(response.text)
        # Persist ordinary JSON-compatible fields. The edge needs only verdict,
        # while the next author call needs rationale and the concrete feedback.
        return {"judge_history": [*history, response], "review": review.model_dump()}

    def route(state: ReviewState):
        """Continue revision only when the draft needs work and budget remains.

        LangGraph calls this after assess with its validated review and round
        count in ``state``. Return ``author`` to revise or ``finish`` to deliver.

        The return value is an edge label, not a model instruction. The mapping
        below translates it to a node. Application code enforces the budget;
        neither agent can request an extra iteration after the limit.
        """
        # Approval wins even on the last allowed round. Budget exhaustion must
        # not relabel a genuinely approved final draft as a failure.
        if state["review"]["verdict"] == "approve":
            return "finish"
        # A failed last review ends honestly as unapproved, not as a success.
        if state["round"] >= max_rounds:
            return "finish"
        # Validation guarantees the remaining verdict is revise. Since capacity
        # remains, send the actual feedback to the next author invocation.
        return "author"

    def finish(state: ReviewState):
        """Deliver the reviewed draft when approval or the revision budget ends the cycle.

        Return a partial state update with the final conversation and outcome;
        LangGraph then follows finish to END and returns state to the caller.
        ``state`` supplies the latest draft, parsed review, and user history.

        Both approval and limit exhaustion arrive here. The verdict distinguishes
        them; the selected edge alone cannot. Keep the latest draft useful, but
        visibly mark it unapproved and retain its unresolved feedback at the limit.
        """
        approved = state["review"]["verdict"] == "approve"
        answer = state["draft"]
        # We can reach finish with revise only when the draft budget is exhausted.
        # Silence here would make a completed execution look like content approval.
        if not approved:
            answer = (
                "Review limit reached — draft NOT approved.\n\n"
                + answer
                + "\n\nOutstanding review:\n"
                + "\n".join(state["review"]["feedback"])
            )
        # AIMessage marks assistant-role text for the external conversation.
        # Constructing it does not call a model; answer is the already-produced
        # draft with a local warning appended when review did not approve it.
        return {
            "messages": [*state["messages"], AIMessage(content=answer)],
            "outcome": "approved" if approved else "limit_reached",
        }

    # StateGraph defines the application control flow explicitly. The functions
    # above are nodes; a model response cannot skip the judge or call finish itself.
    # State updates replace their named fields, leaving other fields untouched.
    # Actual return edge (author and judge each denote a single node):
    #
    # START -> begin -> author -> judge -- approve or budget spent --> finish -> END
    #                    ^         |
    #                    |         | revise, round < max_rounds
    #                    +---------+
    #
    # Every revised draft returns through judge. Approval takes precedence on
    # the last round; exhausting the budget with revise finishes unapproved.
    # Provider or validation errors propagate instead of taking a finish edge.
    #
    graph = StateGraph(ReviewState)
    graph.add_node("begin", begin)
    graph.add_node(
        "author",
        write,
        metadata={
            "report_description": "Write the first draft or revise it from judge feedback."
        },
    )
    graph.add_node(
        "judge",
        assess,
        metadata={
            "report_description": "Validate the judge verdict against the routing contract."
        },
    )
    graph.add_node("finish", finish)
    # START/END are framework sentinels, not functions or additional model calls.
    # Every user invocation must initialize state, draft, and then face review.
    graph.add_edge(START, "begin")
    graph.add_edge("begin", "author")
    graph.add_edge("author", "judge")
    # This is the review loop: only this edge is conditional. route reads the
    # validated judge result and round counter; returning author closes the cycle.
    # There is deliberately no direct author → finish path that bypasses review.
    graph.add_conditional_edges(
        "judge", route, {"author": "author", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    # Compilation returns a runnable graph; it does not execute the first round.
    # The name identifies this workflow in the execution tree and tracing systems.
    return graph.compile(name="review_loop")
