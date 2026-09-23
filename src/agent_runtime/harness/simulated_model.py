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
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from pydantic import PrivateAttr

from agent_runtime.harness.demo_meter import ContextSimulation, message_record


class ScriptedChatModel(FakeMessagesListChatModel):
    """Supply fixed assistant messages while the real graph executes its decisions.

    Construct with responses=[AIMessage(...), ...]. Tool requests must be authored
    in those responses: binding tools does not make this fake model choose them.
    This keeps training runs reproducible without pretending to test an LLM.
    """

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Expose each authored response through the native model streaming API.

        AG-UI reconstructs child transcripts from streamed model events because
        child messages are not in the parent's graph state. The fake model's
        invoke-only fallback omits these events. Complete text and structural tool chunks preserve
        the scripted answer and tool calls without inventing token timing.
        Calling the existing generator once preserves model cursors and metering.
        """
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        response = result.generations[0].message
        # The adapter expects tool names and argument deltas in separate chunks.
        # These are protocol boundaries, not a simulation of token timing.
        if response.content:
            yield ChatGenerationChunk(message=AIMessageChunk(content=response.content))
        for index, call in enumerate(response.tool_calls):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": "",
                            "id": call["id"],
                            "index": index,
                        }
                    ],
                )
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": None,
                            "args": json.dumps(call["args"]),
                            "id": None,
                            "index": index,
                        }
                    ],
                )
            )
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
        # Metadata and usage appear exactly once, preserving the metered response
        # when LangChain combines these chunks into a completed AIMessage.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                **response.model_dump(exclude={"type", "content", "tool_calls"}),
                content="",
            )
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Expose bound schemas to callbacks while scripts own tool choices.

        ``tools`` is converted to the OpenAI-compatible schema consumed by the
        meter and callback metadata. Binding never selects or synthesizes a tool
        call; each scripted response must author that decision explicitly.
        ``tool_choice`` and other options are accepted for framework compatibility
        but ignored here; return a runnable binding, without invoking it.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return self.bind(
            tool_definitions=[convert_to_openai_tool(tool) for tool in tools]
        )

    def _get_ls_params(self, stop=None, **kwargs):
        """Identify this deterministic fixture to LangSmith-compatible callbacks.

        The labels intentionally describe a demo provider/model, preventing a
        report from presenting scripted output as a live provider generation.
        Return callback metadata only; ``stop`` and extra generation settings
        are unused because they cannot alter this fixture's provider identity.
        """
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
        """Include available tools in offline usage so input accounting is complete.

        This stores definitions for metering and returns a callback-visible binding.
        A separate scripted adapter per agent is essential: sharing one would
        overwrite both the available schemas and the sequence of fixed answers.
        The returned binding carries the same definitions into ``_generate`` while
        the model instance keeps one simulation and response cursor per agent.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self._tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tool_definitions=self._tools)

    def _generate(self, messages, *args, **kwargs):
        """Produce a scripted model turn with usage the reporter can inspect.

        LangChain calls this hook with the request transcript in ``messages``;
        remaining arguments pass to its fake-model implementation. The returned
        ChatResult holds a generation containing an AIMessage. Its tool_calls
        are authored proposals for the graph to execute, not tool output.
        Metering advances this instance's simulated context and may reject a
        transcript that dropped earlier messages. No provider is contacted.
        """
        result = super()._generate(messages, *args, **kwargs)
        # FakeMessagesListChatModel can reuse its response objects. Meter a copy
        # so usage from this call cannot mutate the authored scenario fixtures.
        response = result.generations[0].message.model_copy(deep=True)
        response = meter_response(response, self._simulation, self._tools, messages)
        result.generations[0].message = response
        return result


def meter_response(response, simulation, tool_definitions, messages):
    """Let all offline model fixtures report usage under the same counting rules.

    Return the supplied LangChain AIMessage with simulated usage attached.
    ``response`` is copied by callers before metering; ``simulation`` owns the
    append-only context invariant; and ``tool_definitions``/``messages`` are the
    exact request representation to count. The function mutates only the supplied
    response metadata and usage fields, never the authored scenario fixture.
    ``simulation`` belongs to one conversation; ``tool_definitions`` are this
    request bound schemas. Keeping accounting here ensures a shared model and
    older per-agent fixtures use identical token and cost semantics.
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
