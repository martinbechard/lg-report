"""Own a fictional correction scenario and explicitly scripted answer outcomes.

Offline model decisions demonstrate tool execution and context mechanics only.
Both modes answer correctly in this fixture: we do not manufacture model failure
by hard-coding a wrong naive answer. Live mode reuses only USER_PROMPTS and lets
the configured model choose actions; only live runs can reveal model confusion.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.demo_meter import TOKEN_ESTIMATE_BASIS, ContextSimulation
from agent_runtime.harness.model_factory import client_prompts, model_responses
from agent_runtime.harness.simulated_model import MeteredDemoModel

CORRECTED_DESCRIPTION = "The laptop screen cracked when it fell from a desk at home."
# mode marks the optional reread; it does not change user input order.
CONVERSATION = [
    {
        "role": "client",
        "content": "What deductible does policy POL-001 apply to accidental laptop damage?",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "read_policy",
                "args": {},
                "id": "policy-read",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_policy",
        "tool_call_id": "policy-read",
        "content": "Expected: actual read_policy result for {}; supplied by the running tool.",
    },
    {"role": "ai", "content": "The policy deductible is CAD 150."},
    {
        "role": "client",
        "content": "What happened in claim CLM-001, and what is its status?",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "read_claim",
                "args": {},
                "id": "initial-read",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_claim",
        "tool_call_id": "initial-read",
        "content": "Expected: actual read_claim result for {}; supplied by the running tool.",
    },
    {
        "role": "ai",
        "content": "The laptop was stolen from a parked car. Status: pending.",
    },
    {
        "role": "client",
        "content": "Correct the claim: the laptop was not stolen. Its screen cracked when it fell from a desk at "
        "home. Replace the description with 'The laptop screen cracked when it fell from a desk at "
        "home.' and change the status to approved.",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_claim",
                "args": {
                    "expected_revision": 1,
                    "description": "The laptop screen cracked when it fell from a desk at home.",
                    "status": "approved",
                },
                "id": "correction",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_claim",
        "tool_call_id": "correction",
        "content": "Expected: actual edit_claim result for {'expected_revision': 1, 'description': 'The laptop "
        "screen cracked when it fell from a desk at home.', 'status': 'approved'}; supplied by the "
        "running tool.",
    },
    {"role": "ai", "content": "The correction was saved."},
    {"role": "client", "content": "What is the policy's coverage limit?"},
    {"role": "ai", "content": "The policy coverage limit is CAD 2,000."},
    {
        "role": "client",
        "content": "What happened to the laptop, where did it happen, and what is the current claim status?",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "read_claim",
                "args": {},
                "id": "follow-up-read",
            }
        ],
        "mode": "managed",
    },
    {
        "role": "tool",
        "name": "read_claim",
        "tool_call_id": "follow-up-read",
        "content": "Expected: actual read_claim result for {}; supplied by the running tool.",
        "mode": "managed",
    },
    {
        "role": "ai",
        "content": "The laptop screen cracked when it fell from a desk at home. The status is approved.",
    },
]
USER_PROMPTS = client_prompts(CONVERSATION)


class UncachedDemoModel(MeteredDemoModel):
    """Meter each request independently because this lesson deletes old context.

    The shared meter requires an append-only prefix. Starting a new simulation
    for EVERY call in BOTH modes avoids changing that invariant or inventing
    provider cache behavior. Sizes are illustrative; no cache reuse is modeled.
    """

    def _generate(self, messages, *args, **kwargs):
        """Count the actual current input without retaining a previous cache ledger."""
        # The shared simulator assumes append-only history. Reset its counting
        # ledger for both modes so legitimate context pruning cannot trigger that
        # unrelated invariant or be misrepresented as a provider cache hit.
        self._simulation = ContextSimulation()
        result = super()._generate(messages, *args, **kwargs)
        result.generations[0].message.response_metadata["usage_basis"] = (
            f"{TOKEN_ESTIMATE_BASIS}; no cache reuse modeled"
        )
        return result


def make_simulated_model(mode):
    """Replay agent tool choices to inspect context, not to measure answer quality.

    Both modes produce the same correct final answer. The naive fixture combines
    the original snapshot and successful edits; the managed fixture requests a
    fresh read because its claim snapshot was removed. All choices are scripted.
    """
    conversation = [entry for entry in CONVERSATION if entry.get("mode", mode) == mode]
    return UncachedDemoModel(responses=model_responses(conversation))


def build_scripted_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model(options.get("mode", "naive"))}
