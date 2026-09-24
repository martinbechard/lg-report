"""Feed the real nested graphs independent, explicitly simulated role decisions.

Every request is metered independently: a new reviewer context must not be
mistaken for an append-only conversation or imply cache reuse. This accounting
choice neither changes nor shortens any message history. Queues fail when
exhausted instead of silently cycling back to an earlier approval.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.messages import AIMessage
from pydantic import PrivateAttr

from agent_runtime.harness.demo_meter import ContextSimulation
from agent_runtime.harness.simulated_model import MeteredDemoModel
from samples.nested_workflows.scenarios import USER_PROMPTS as USER_PROMPTS, script


class NestedDemoModel(MeteredDemoModel):
    """Keep one role's response cursor while recording each actual prompt for tests."""

    _issued: int = PrivateAttr(default=0)
    _seen: list = PrivateAttr(default_factory=list)

    @property
    def requests(self):
        """Expose defensive copies of observed inputs, not authored expectations."""
        return [[message.model_copy(deep=True) for message in messages] for messages in self._seen]

    def _generate(self, messages, *args, **kwargs):
        if self._issued >= len(self.responses):
            raise RuntimeError("The scripted role response queue is exhausted")
        self._seen.append([message.model_copy(deep=True) for message in messages])
        self._issued += 1
        self._simulation = ContextSimulation()
        result = super()._generate(messages, *args, **kwargs)
        result.generations[0].message.response_metadata["usage_basis"] = (
            "Simulated canonical-JSON tokens per actual request; no cache reuse assumed"
        )
        return result


def build_models(options):
    """Supply fresh named adapters in the existing SampleCatalog factory protocol."""
    return {
        name: NestedDemoModel(
            responses=[AIMessage(content=json.dumps(item)) for item in responses],
            metadata={"report_description": f"Scripted nested-workflow {name} decisions."},
        )
        for name, responses in script(options).items()
    }
