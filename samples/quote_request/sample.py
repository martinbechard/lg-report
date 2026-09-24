"""Author the quote request example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

The initial request, AI clarification decisions, and human answers share one
ordered script. QuoteDecision is structured output, not an executing tool. The
real workflow validates decisions and pauses for answers; only choices are
scripted. This demonstrates routing, not live semantic judgment.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.messages import AIMessage

from agent_runtime.harness.model_factory import client_prompts, model_responses
from agent_runtime.harness.simulated_model import ScriptedChatModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "quote_request",
    "name": "Quote clarification",
    "description": "Answer the agent’s clarification questions.",
    "interaction": "clarification",
}

# QuoteDecision is structured model output validated by the workflow, not an
# executing tool. Its ask text becomes an interrupt; the next human entry resumes
# that same turn. Only the initial client entry starts a new request.
CONVERSATION = [
    {
        "role": "client",
        "content": json.dumps(
            {
                "name": "Alex",
                "email": "alex@example.com",
                "service": "Print 500 individually addressed invitations. Use the exact same text on "
                "every copy; no variable printing.",
                "quantity": 500,
                "recipients": "650 guests, each must receive their own invitation.",
                "delivery": "One batch to our office; we will mail them ourselves.",
            }
        ),
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "QuoteDecision",
                "args": {
                    "action": "ask",
                    "reason": "Individual addressing conflicts with identical copies and no variable "
                    "printing.",
                    "text": "Should each invitation be personalized, or should only the envelopes "
                    "carry individual names and addresses?",
                },
                "id": "decision-0",
            }
        ],
    },
    {
        "role": "human",
        "interaction": "clarification",
        "content": "The invitation text should be identical. Put each guest's name and address on the envelopes "
        "instead.",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "QuoteDecision",
                "args": {
                    "action": "ask",
                    "reason": "500 invitations cannot provide one invitation each to 650 guests.",
                    "text": "You requested 500 invitations for 650 guests, each receiving their own. "
                    "Should we increase the quantity, or are invitations shared by "
                    "households?",
                },
                "id": "decision-1",
            }
        ],
    },
    {
        "role": "human",
        "interaction": "clarification",
        "content": "There are 500 households among the 650 guests. Send one invitation per household.",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "QuoteDecision",
                "args": {
                    "action": "ask",
                    "reason": "Envelope printing changes the scope of the quote.",
                    "text": "Should the quote include 500 individually printed envelopes, and will "
                    "you supply the address list?",
                },
                "id": "decision-2",
            }
        ],
    },
    {
        "role": "human",
        "interaction": "clarification",
        "content": "Include 500 printed envelopes. We'll supply a spreadsheet of names and addresses.",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "QuoteDecision",
                "args": {
                    "action": "complete",
                    "reason": "The human clarified where personalization belongs and who supplies "
                    "addresses.",
                    "text": "Prepare a request for 500 identical invitations, one per household, plus "
                    "500 individually addressed envelopes using the customer's spreadsheet. "
                    "Deliver one batch to the office for the customer to mail. Supplier "
                    "pricing and acceptance remain pending.",
                },
                "id": "decision-3",
            }
        ],
    },
]

USER_PROMPTS = client_prompts(CONVERSATION)
INITIAL_VALUES = json.loads(USER_PROMPTS[0])
ANSWERS = [entry["content"] for entry in CONVERSATION if entry["role"] == "human"]
DECISIONS = [message.tool_calls[0]["args"] for message in model_responses(CONVERSATION)]


def make_simulated_model(decisions=None):
    """Extract the quote replies, or accept replacement decisions for routing tests.

    Preserve the unmetered adapter: missing provider usage remains unavailable.
    The workflow validates each QuoteDecision and pauses for real or scripted
    human answers; this function neither executes tools nor resumes interrupts.
    """
    responses = (
        model_responses(CONVERSATION)
        if decisions is None
        else [
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
            for index, decision in enumerate(decisions)
        ]
    )
    return ScriptedChatModel(responses=responses)


def build_scripted_models(options):
    """Keep quote decisions unmetered through the catalog's optional callback.

    options contains sample defaults merged with run overrides. This fixed story
    ignores them. The workflow caller receives a fresh model; live mode bypasses
    this callback entirely.
    """
    return {"workflow": make_simulated_model()}
