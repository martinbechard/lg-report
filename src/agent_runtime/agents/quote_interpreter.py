"""Interpret a quote request and its clarification history through a native agent.

The agent owns instructions and context formatting. LangChain applies workflow
middleware and validates the structured decision. The workflow supplies request data and human answers,
then owns checkpoints, routing, and human interrupts. No quote is sent to a supplier.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
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


class QuoteAssessment(TypedDict):
    """Public input for one assessment, independent of the model's message protocol."""

    initial_request: dict
    conversation: list[
        dict
    ]  # Ordered human question/answer pairs retained by the harness.


def input_context(request=None):
    """Expose role-owned input assembly without requesting a model decision.

    None means a new request has not been supplied, so only stable instructions
    and the response schema are known. A clarification retains its prior data.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if request is not None:
        messages.append(HumanMessage(content=json.dumps(request)))
    return {"messages": messages, "tools": [convert_to_openai_tool(QuoteDecision)]}


def build_agent(parameters: dict):
    """Return a named runnable that maps QuoteAssessment to a validated QuoteDecision.

    The workflow receives a validated decision, independent of tool execution
    and structured-output plumbing. Malformed decisions propagate as errors;
    the library does not retry schema errors for this role.
    """
    decider = create_agent(
        **parameters,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(QuoteDecision, handle_errors=False),
    )

    def assess(request: QuoteAssessment, config: RunnableConfig) -> QuoteDecision:
        """Interpret all supplied context and reject invalid model decisions."""
        # The harness chooses which original values and answers to retain. The
        # agent serializes that context into its own model-facing representation,
        # separating user evidence from the stable role instructions.
        result = decider.invoke(
            {"messages": input_context(request)["messages"][1:]},
            config=config,
        )
        # ToolStrategy validates the decision and returns the Pydantic object.
        # A plain text answer is not a routing decision and must not end the flow.
        if result.get("structured_response") is None:
            raise ValueError("Expected exactly one QuoteDecision from the model")
        return result["structured_response"]

    return RunnableLambda(assess).with_config(
        run_name="quote_interpreter",
        metadata={
            "report_description": "Interpret quote scope and request human clarification when needed."
        },
    )
