"""Own a fictional correction scenario and explicitly scripted answer outcomes.

Offline model decisions demonstrate tool execution and context mechanics only.
Both modes answer correctly in this fixture: we do not manufacture model failure
by hard-coding a wrong naive answer. Live mode reuses only USER_PROMPTS and lets
the configured model choose actions; only live runs can reveal model confusion.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from agent_runtime.harness.demo_meter import ContextSimulation
from agent_runtime.harness.simulated_model import MeteredDemoModel

CORRECTED_DESCRIPTION = "The laptop screen cracked when it fell from a desk at home."
USER_PROMPTS = [
    "What deductible does policy POL-001 apply to accidental laptop damage?",
    "What happened in claim CLM-001, and what is its status?",
    (
        "Correct the claim: the laptop was not stolen. Its screen cracked when it "
        "fell from a desk at home. Replace the description with "
        f"'{CORRECTED_DESCRIPTION}' and change the status to approved."
    ),
    "What is the policy's coverage limit?",
    "What happened to the laptop, where did it happen, and what is the current claim status?",
]


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
            "Simulated canonical-JSON words/punctuation; no cache reuse modeled"
        )
        return result


def make_simulated_model(mode):
    """Replay agent tool choices to inspect context, not to measure answer quality.

    Both modes produce the same correct final answer. The naive fixture combines
    the original snapshot and successful edits; the managed fixture requests a
    fresh read because its claim snapshot was removed. All choices are scripted.
    """
    answer = f"{CORRECTED_DESCRIPTION} The status is approved."
    # These are model outputs, not workflow actions. The real agent graph still
    # routes each tool request, executes the tool, and feeds its actual result
    # into the next model call. Live mode never imports these decisions.
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_policy",
                    "args": {},
                    "id": "policy-read",
                }
            ],
        ),
        AIMessage(content="The policy deductible is CAD 150."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_claim",
                    "args": {},
                    "id": "initial-read",
                }
            ],
        ),
        AIMessage(content="The laptop was stolen from a parked car. Status: pending."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "edit_claim",
                    "id": "correction",
                    "args": {
                        "expected_revision": 1,
                        "description": CORRECTED_DESCRIPTION,
                        "status": "approved",
                    },
                }
            ],
        ),
        AIMessage(content="The correction was saved."),
        # This policy-only follow-up deliberately makes no claim read, including
        # in managed mode. A human query, not a pending-reload flag, drives tools.
        AIMessage(content="The policy coverage limit is CAD 2,000."),
    ]
    if mode == "managed":
        # Like every offline decision, this is an authored model tool request.
        # The workflow does not insert it; live mode leaves selection to the LLM.
        responses.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_claim",
                        "args": {},
                        "id": "follow-up-read",
                    }
                ],
            )
        )
    responses.append(AIMessage(content=answer))
    return UncachedDemoModel(responses=responses)


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model(options.get("mode", "naive"))}
