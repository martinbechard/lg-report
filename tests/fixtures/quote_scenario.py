"""Provide test projections and alternate decisions for the quote scenario.

The sample owns its chronological conversation. Tests derive expected values
here and can substitute decisions to exercise routing; these helpers are not
part of sample discovery or the model factory used by application clients.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from agent_runtime.harness.simulated_model import SimulatedModel
from samples.quote_request.sample import CONVERSATION, build_scripted_models

INITIAL_VALUES = json.loads(CONVERSATION[0]["content"])
ANSWERS = [entry["content"] for entry in CONVERSATION if entry["role"] == "human"]
DECISIONS = [
    entry["tool_calls"][0]["args"]
    for entry in CONVERSATION
    if entry["role"] == "quote_interpreter"
]


def make_simulated_model(decisions=None):
    """Use production demo settings, optionally replacing only the test decisions."""
    if decisions is None:
        return build_scripted_models({})["workflow"]
    return SimulatedModel(
        cache_reuse=False,
        conversation=[
            {
                "role": "quote_interpreter",
                "content": "",
                "tool_calls": [
                    {
                        "name": "QuoteDecision",
                        "args": decision,
                        "id": f"decision-{index}",
                    }
                ],
            }
            for index, decision in enumerate(decisions)
        ],
    )
