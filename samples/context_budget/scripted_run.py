"""Supply deterministic requests and decisions for the context-budget lesson.

The real middleware chooses when to summarize and the real task tool runs the
child. Summary text and decisions are scripted, so this verifies mechanics,
not live summarization quality. Usage is illustrative and assumes no cache hits.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from agent_runtime.harness.demo_meter import ContextSimulation
from agent_runtime.harness.simulated_model import MeteredDemoModel


class CompactingDemoModel(MeteredDemoModel):
    """Meter each rewritten request independently without claiming cache reuse."""

    def _generate(self, messages, *args, **kwargs):
        # The ordinary demo meter enforces append-only conversations. Compaction
        # deliberately violates that assumption; reset only this fixture's meter.
        self._simulation = ContextSimulation()
        result = super()._generate(messages, *args, **kwargs)
        result.generations[0].message.response_metadata["usage_basis"] = (
            "Simulated canonical-JSON tokens; rewritten context, no cache reuse assumed"
        )
        return result


# Repetition is an intentional load fixture, not additional factual evidence.
# The marker lets tests prove raw older input was actually removed at the peer
# boundary, while a summary preserves the meaningful language constraint.
USER_PROMPTS = [
    "PARENT_ONLY_DETAIL: Answer in French. Explain agent workflow delegation. "
    + "This is repeated background for the compaction demonstration. " * 35,
    "Now explain how the specialist's tool observations return to its parent.",
]
PEER_BRIEF = "Preserve the French language requirement; obtain a specialist echo summary and explain delegation."
ASSIGNMENT = (
    "Echo the assigned delegation text using echo_tool, then summarize the result."
)
WORKFLOW_SUMMARY = "The user requires French answers about workflow delegation. Preserve that constraint."
CHILD_SUMMARY = "The assignment is to explain delegation. The echo repeats supplied text without adding evidence."
CHILD_ANSWER = "An agent selects a tool, observes its result, and uses it to answer."
FINAL_ANSWER = (
    "Le spécialiste reçoit une mission autonome et renvoie sa réponse au parent."
)


def _model(responses, description):
    """Keep each role's response cursor independent and label trace intent."""
    return CompactingDemoModel(
        responses=responses, metadata={"report_description": description}
    )


def make_simulated_models():
    """Return models in the workflow factory's declared dependency order."""
    parent_responses, child_responses = [], []
    for turn in range(len(USER_PROMPTS)):
        parent_responses.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "subagent_type": "workflow-specialist",
                                "description": ASSIGNMENT,
                            },
                            "id": f"delegate-{turn}",
                        }
                    ],
                ),
                AIMessage(content=FINAL_ANSWER),
            ]
        )
        # The first lookup produces a deliberately large actual observation:
        # echo_tool echoes text. A second lookup gives middleware an
        # older complete tool exchange to compact without splitting a call/result.
        for lookup, topic in enumerate(
            [
                "CHILD_ONLY_DETAIL: " + "delegation reference background " * 45,
                "tool observations",
            ]
        ):
            child_responses.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo_tool",
                            "args": {"text": topic},
                            "id": f"lookup-{turn}-{lookup}",
                        }
                    ],
                )
            )
        child_responses.append(AIMessage(content=CHILD_ANSWER))
    return (
        _model(
            [AIMessage(content=PEER_BRIEF)] * len(USER_PROMPTS),
            "Plan from shared workflow history.",
        ),
        _model(parent_responses, "Delegate from shared history and respond."),
        _model(child_responses, "Research within isolated subagent history."),
        _model(
            [AIMessage(content=WORKFLOW_SUMMARY)], "Summarize shared workflow history."
        ),
        _model(
            [AIMessage(content=CHILD_SUMMARY)], "Summarize isolated subagent history."
        ),
    )


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    names = (
        "planner",
        "responder",
        "workflow-specialist",
        "workflow-summary",
        "subagent-summary",
    )
    return dict(zip(names, make_simulated_models(), strict=True))
