"""Interpret a quote request and its clarification history using one model call.

The agent owns the stable role instructions and decision-tool binding. The
workflow supplies conversation messages, validates the returned tool call, and
owns checkpoints, routing, and human interrupts. No quote is sent to a supplier.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, Field


class QuoteDecision(BaseModel):
    """Strict routing payload returned by the quote-interpreter model.

    ``action`` is the only field the graph uses for control flow.  ``reason``
    gives the human context for a question, while ``text`` is either that
    question or the model's final local scope summary.  The minimum lengths
    reject empty tool arguments; they do not decide whether a customer's
    business request is substantively complete.  That judgment remains with the
    model under ``SYSTEM_PROMPT``.
    """

    # Keeping the action as a Literal makes unexpected model routes fail visibly
    # instead of falling through to an unplanned graph edge.
    action: Literal["ask", "complete"]
    # A non-empty explanation is required so a human can understand why a pause
    # occurred; it is still model-authored content and is not a business rule.
    reason: str = Field(min_length=1, description="Explain the decision to the human.")
    # The same field carries the focused question or completed local summary,
    # depending on the selected action.
    text: str = Field(
        min_length=1,
        description="A specific clarification question for ask, or the agreed request summary for complete.",
    )


SYSTEM_PROMPT = """You help a human prepare a quote request. Read the entire request
and clarification conversation. Decide whether its meaning is clear enough to
prepare an accurate request. Notice conflicting requirements, implausible scope,
ambiguous units, and assumptions that would materially change the quote. Ask the
human a focused question in your own words when their judgment or information is
needed. Check for multiple independent issues across the whole request, not just
the first inconsistency. Keep all unresolved issues in view after every answer.
You may group closely related questions; do not complete while a material issue
remains unresolved. Explain the concrete uncertainty. This is a conversation, not a required
field checklist. Do not silently repair contradictions or invent requirements,
prices, feasibility guarantees, or human answers. Reconsider the request after
every answer; an unclear answer may need another question. Complete without a
question when the request is already coherent. Completion summarizes the agreed
scope and any explicitly accepted uncertainty; it does not send a quote or
promise supplier acceptance. Return exactly one QuoteDecision tool call.
"""


def build_agent(model):
    """Prepare a scope interpreter so the workflow can decide when to ask the human.

    The supplied model may be a live provider or an offline fixture. Binding the
    decision schema exposes the output contract; the workflow still checks that
    the response contains exactly one valid decision before choosing an edge.

    ``model`` must support bind_tools and invoke. Return a LangChain runnable
    accepting a list of messages and execution config, and yielding the model's
    AIMessage. This wrapper is not a DeepAgents tool-execution loop.
    """
    # Binding advertises QuoteDecision's Pydantic schema as a tool definition
    # and asks the model to output a call to it. It does not execute Python or
    # validate the eventual arguments. The workflow reads and validates those
    # arguments itself; there is no tool result sent back to the model here.
    decider = model.bind_tools([QuoteDecision], tool_choice="QuoteDecision")

    def assess(messages, config: RunnableConfig):
        """Reassess the request so unresolved scope can become a question or summary.

        Prepending instructions on each call keeps them out of stored human
        history. Forward configuration so callbacks retain usage and trace
        context. Return the original response and propagate provider failures;
        the workflow owns validation and must not route on invented output.

        ``messages`` is the workflow's list of user-context messages for this
        assessment. ``config`` carries LangChain execution settings, including
        callbacks used for reporting. Return the model's AIMessage unchanged,
        including proposed tool_calls, content and any usage metadata.
        """
        # SystemMessage identifies role/behavior instructions to the chat model;
        # HumanMessage identifies user-supplied context. These are LangChain
        # message containers, not DeepAgents-specific operations. Supplying
        # both roles lets the model distinguish instructions from request data.
        # One invoke performs one model invocation; no tools are dispatched here.
        return decider.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *messages], config=config
        )

    # A named role makes the agent boundary visible in execution reports without
    # adding another model call or taking ownership of workflow persistence.
    # RunnableLambda adapts this named Python function to LangChain's invoke
    # interface; despite its name, it does not require a Python lambda. The
    # wrapper forwards config/callbacks and labels the run for our report.
    return RunnableLambda(assess).with_config(
        run_name="quote_interpreter",
        metadata={
            "report_description": "Interpret quote scope and request human clarification when needed."
        },
    )
