"""Supply a complete but contradictory request and an explicitly scripted demo.

These authored decisions test wiring, not an LLM's ability to understand meaning.
Live mode replaces this model so questions arise from actual model judgment.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# LangChain's AIMessage holds an assistant response; constructing it runs nothing.
# Its tool_calls, when present, are proposed names/arguments, not tool results.
from langchain_core.messages import AIMessage

from agent_runtime.harness.simulated_model import ScriptedChatModel

# The initial request is intentionally internally inconsistent. It gives a live
# model reasons to ask clarifying questions and gives the offline fixture a
# stable demonstration of conflict detection without requiring a provider.
INITIAL_VALUES = {
    "name": "Alex",
    "email": "alex@example.com",
    "service": "Print 500 individually addressed invitations. Use the exact same text on every copy; no variable printing.",
    "quantity": 500,
    "recipients": "650 guests, each must receive their own invitation.",
    "delivery": "One batch to our office; we will mail them ourselves.",
}
ANSWERS = [
    "The invitation text should be identical. Put each guest's name and address on the envelopes instead.",
    "There are 500 households among the 650 guests. Send one invitation per household.",
    "Include 500 printed envelopes. We'll supply a spreadsheet of names and addresses.",
]
DECISIONS = [
    {
        "action": "ask",
        "reason": "Individual addressing conflicts with identical copies and no variable printing.",
        "text": "Should each invitation be personalized, or should only the envelopes carry individual names and addresses?",
    },
    {
        "action": "ask",
        "reason": "500 invitations cannot provide one invitation each to 650 guests.",
        "text": "You requested 500 invitations for 650 guests, each receiving their own. Should we increase the quantity, or are invitations shared by households?",
    },
    {
        "action": "ask",
        "reason": "Envelope printing changes the scope of the quote.",
        "text": "Should the quote include 500 individually printed envelopes, and will you supply the address list?",
    },
    {
        "action": "complete",
        "reason": "The human clarified where personalization belongs and who supplies addresses.",
        "text": "Prepare a request for 500 identical invitations, one per household, plus 500 individually addressed envelopes using the customer's spreadsheet. Deliver one batch to the office for the customer to mail. Supplier pricing and acceptance remain pending.",
    },
]


def make_simulated_model(decisions=None):
    """Replay authored quote decisions without fabricating provider usage.

    Args:
        decisions: Optional replacement sequence of tool-call argument mappings.
            When omitted, the canonical four-step fixture is used.

    Returns:
        A scripted model that emits one ``QuoteDecision`` call per mapping.

    Side effects:
        Construction performs no model or network I/O. The workflow invokes
        the model later and validates the proposed decision arguments; no
        QuoteDecision tool executes. The client supplies answers between asks.
    """
    # The simulator emits structured tool calls so the real workflow validates
    # action/reason/text routing. It does not execute decisions or answer the
    # user itself; the client remains the boundary for human responses.
    return ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "QuoteDecision",
                        "args": decision,
                        "id": f"decision-{index}",
                    }
                ],
            )
            for index, decision in enumerate(
                DECISIONS if decisions is None else decisions
            )
        ]
    )


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}


# The initial structured request is still ordinary client input.
import json

USER_PROMPTS = [json.dumps(INITIAL_VALUES)]
