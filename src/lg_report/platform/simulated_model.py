"""Replace provider calls with scripted answers while executing the real agent graph.

ScriptedChatModel supplies authored assistant messages, including any tool-call
requests. LangGraph still executes those tools; this module does not fabricate
their results. MeteredDemoModel adds usage computed from the actual messages and
bound tool definitions using demo_meter's deterministic counting rules.

MeteredDemoModel is a per-agent sequential fixture. SharedSimulatedModel supports
the single-LLM dispatcher sample with isolated scripts and accounting per binding. Counts and reasoning fixtures are
illustrative: they make cost-report examples reproducible without provider fees,
but do not predict a real model's tokenizer, reasoning, or caching behavior.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from pydantic import PrivateAttr

from lg_report.platform.demo_meter import ContextSimulation, message_record


class ScriptedChatModel(FakeMessagesListChatModel):
    """Supply fixed assistant messages while the real graph executes its decisions.

    Construct with responses=[AIMessage(...), ...]. Tool requests must be authored
    in those responses: binding tools does not make this fake model choose them.
    This keeps training runs reproducible without pretending to test an LLM.
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Expose bound schemas to callbacks; scripted responses own tool choices."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return self.bind(
            tool_definitions=[convert_to_openai_tool(tool) for tool in tools]
        )

    def _get_ls_params(self, stop=None, **kwargs):
        return {
            "ls_provider": "demo",
            "ls_model_name": "scripted-chat",
            "ls_model_type": "chat",
        }


class MeteredDemoModel(ScriptedChatModel):
    """Attach deterministic usage to scripted messages for cost-analysis training.

    Construct one instance per agent, with its own responses list. Reusing the
    parent's instance for a subagent would mix response cursors and independent
    caches. The usage shape matches LangChain so capture/exporters need no separate
    accounting path for demos; actual provider calls never use this estimator.
    """

    _simulation: ContextSimulation = PrivateAttr(default_factory=ContextSimulation)
    _tools: list[dict] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Retain tool schemas for accounting; the fixture still chooses tool calls.

        This stores definitions for metering and returns a callback-visible binding.
        A separate scripted adapter per agent is essential: sharing one would
        overwrite both the available schemas and the sequence of fixed answers.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self._tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tool_definitions=self._tools)

    def _generate(self, messages, *args, **kwargs):
        result = super()._generate(messages, *args, **kwargs)
        # FakeMessagesListChatModel can reuse its response objects. Meter a copy
        # so usage from this call cannot mutate the authored scenario fixtures.
        response = result.generations[0].message.model_copy(deep=True)
        response = meter_response(response, self._simulation, self._tools, messages)
        result.generations[0].message = response
        return result


def meter_response(response, simulation, tool_definitions, messages):
    """Attach the common simulated usage shape to an already-copied response.

    simulation belongs to one conversation; tool_definitions are this request's
    bound schemas. Keeping accounting here ensures a shared model and older
    per-agent fixtures use identical token and cost semantics.
    """
    usage_entry, role_token_counts = simulation.record(
        tool_definitions,
        [message_record(message) for message in messages],
        message_record(response),
    )
    # Reasoning contributes to billed output, but its hidden token allowance
    # is not appended to the visible conversation that later requests reuse.
    reasoning_tokens = response.response_metadata.get("simulated_reasoning_tokens", 0)
    response.usage_metadata = {
        "input_tokens": usage_entry["request_tokens"],
        "output_tokens": usage_entry["response_tokens"] + reasoning_tokens,
        "total_tokens": usage_entry["request_tokens"]
        + usage_entry["response_tokens"]
        + reasoning_tokens,
        "output_token_details": {"reasoning": reasoning_tokens},
        "input_token_details": {"cache_read": usage_entry["cache_read_tokens"]},
    }
    response.response_metadata.update(
        {
            "demo_meter": {
                **{
                    f"simulated_{role}_tokens": count
                    for role, count in role_token_counts.items()
                },
                "context_ledger": json.dumps(usage_entry),
            },
            "usage_basis": "Simulated canonical-JSON words/punctuation; completed conversation is cached",
        }
    )
    return response
