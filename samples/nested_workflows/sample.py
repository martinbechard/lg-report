"""Feed the real nested graphs independent, explicitly simulated role decisions.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

Every request is metered independently: a new reviewer context must not be
mistaken for an append-only conversation or imply cache reuse. This accounting
choice neither changes nor shortens any message history. Queues fail when
exhausted instead of silently cycling back to an earlier approval.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pydantic import PrivateAttr

from agent_runtime.harness.simulated_model import SimulatedModel
from samples.nested_workflows.scenarios import (
    CONVERSATION as CONVERSATION,  # noqa: PLC0414 - chronological catalog export
)
from samples.nested_workflows.scenarios import (
    USER_PROMPTS as USER_PROMPTS,  # noqa: PLC0414 - catalog export
)
from samples.nested_workflows.scenarios import (
    conversation,
)

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "nested_workflows",
    "name": "Nested workflows and context scopes",
    "description": "Planner/tester share outer context; supervisor/coder share a nested "
    "context; each review is isolated. Includes review repair and "
    "tester-driven re-entry.",
    "options": {"scenario": "rework", "max_review_rounds": 3, "max_coding_cycles": 3},
}


class NestedDemoModel(SimulatedModel):
    """Keep one role's response cursor while recording each actual prompt for tests."""

    _seen: list = PrivateAttr(default_factory=list)

    @property
    def requests(self):
        """Expose defensive copies of observed inputs, not authored expectations."""
        return [
            [message.model_copy(deep=True) for message in messages]
            for messages in self._seen
        ]

    def _generate(self, messages, *args, **kwargs):
        """Record actual input before the common simulator selects and meters a reply."""
        self._seen.append([message.model_copy(deep=True) for message in messages])
        return super()._generate(messages, *args, **kwargs)


def build_scripted_models(options):
    """Extract named replies from the selected chronological scenario.

    options merges sample defaults and run overrides. scenario selects the story;
    max_review_rounds and max_coding_cycles bound its scripted retries. The catalog
    uses this callback to preserve the independent metering and exhaustion checks.
    """
    steps = conversation(options)
    return {
        name: NestedDemoModel(
            conversation=steps, cache_reuse=False,
            metadata={
                "report_description": f"Scripted nested-workflow {name} decisions."
            },
        )
        for name in ("planner", "coding_supervisor", "coder", "reviewer", "tester")
    }
