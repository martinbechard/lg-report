"""Replace provider calls with scripted answers while executing the real agent graph.

SimulatedModel consumes one chronological scenario and filters replies by the
native agent name. Shared and solo execution use the same response selection and
per-agent accounting. Tools still execute in the real graph. ScriptedChatModel
remains an unmetered low-level test fixture and supplies streaming mechanics.
Counts and reasoning fixtures are illustrative, not provider charges.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from copy import deepcopy
from threading import Lock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import ensure_config
from pydantic import Field, PrivateAttr, model_validator

from agent_runtime.harness.demo_meter import (
    TOKEN_ESTIMATE_BASIS,
    ContextSimulation,
    message_record,
)


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


class SimulatedModel(ScriptedChatModel):
    """Select each named agent's replies from one chronological sample scenario.

    Construct with conversation=[{"role": "agent_name", "content": "..."}, ...].
    LangChain supplies lc_agent_name from the native agent name; client, human,
    and tool entries remain scenario evidence, never executed model responses.
    Each agent has an independent cursor and cache, even on one shared model.
    agent_name is reserved for direct model clients such as summarization, which
    have no agent identity of their own. No prompt/tool-name inference is used.
    """

    conversation: list[dict]
    agent_name: str | None = None
    # Rewritten or isolated histories cannot promise append-only cache reuse.
    # This option affects accounting only; it never changes graph-owned history.
    cache_reuse: bool = True
    # The unmetered base supplies streaming/tool binding, not response selection.
    responses: list = Field(default_factory=list, exclude=True, repr=False)
    # Positions count only successfully metered responses for the matching agent.
    _positions: dict[str, int] = PrivateAttr(default_factory=dict)
    # Cache state is local to this scenario run, never shared between model instances.
    _contexts: dict[str, ContextSimulation] = PrivateAttr(default_factory=dict)
    # Parallel graph branches must select and advance their own entries atomically.
    _lock: Lock = PrivateAttr(default_factory=Lock)

    @model_validator(mode="before")
    @classmethod
    def copy_scenario(cls, values):
        """Own scenario data per model and reject the superseded response-list API."""
        if "responses" in values:
            raise ValueError("SimulatedModel requires one conversation, not responses")
        return (
            {**values, "conversation": deepcopy(values["conversation"])}
            if "conversation" in values
            else values
        )

    def _next_response(self, agent_name, messages, position):
        """Copy the next matching entry; exhaustion must not replay an old answer.

        Sample fixtures may override this hook to fill an authored template from
        real tool observations. The common generator still owns locking and usage.
        """
        entries = [entry for entry in self.conversation if entry["role"] == agent_name]
        if position >= len(entries):
            raise ValueError(f"No simulated response left for agent {agent_name!r}")
        entry = entries[position]
        return AIMessage(
            content=deepcopy(entry["content"]),
            tool_calls=deepcopy(entry.get("tool_calls", [])),
            response_metadata=deepcopy(entry.get("response_metadata", {})),
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Resolve identity, select one response, and meter this actual request.

        Binding-specific schemas arrive in kwargs, so sharing a model cannot
        overwrite another agent's tools. Advance the cursor only after successful
        metering; failed prefix checks leave the scenario response available.
        """
        # LangChain's event-stream path omits run_manager when calling _stream.
        # Native graph config still carries the same metadata through its context.
        metadata = run_manager.metadata if run_manager else ensure_config()["metadata"]
        agent_name = self.agent_name or (metadata or {}).get("lc_agent_name")
        if not agent_name or agent_name in {"client", "tool", "human", "middleware"}:
            raise ValueError("SimulatedModel requires a named agent (lc_agent_name)")
        if not any(entry["role"] == agent_name for entry in self.conversation):
            raise ValueError(f"No scenario entries for agent {agent_name!r}")
        with self._lock:
            position = self._positions.get(agent_name, 0)
            response = self._next_response(agent_name, messages, position)
            simulation = (
                self._contexts.setdefault(agent_name, ContextSimulation())
                if self.cache_reuse
                else ContextSimulation()
            )
            response = meter_response(
                response, simulation, kwargs.get("tool_definitions", []), messages
            )
            if not self.cache_reuse:
                response.response_metadata["usage_basis"] = (
                    f"{TOKEN_ESTIMATE_BASIS}; per actual request; no cache reuse assumed"
                )
            self._positions[agent_name] = position + 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def meter_response(response, simulation, tool_definitions, messages):
    """Let all offline model fixtures report usage under the same counting rules.

    Return the supplied LangChain AIMessage with simulated usage attached.
    ``response`` is copied by callers before metering; ``simulation`` owns the
    append-only context invariant; and ``tool_definitions``/``messages`` are the
    exact request representation to count. The function mutates only the supplied
    response metadata and usage fields, never the authored scenario fixture.
    ``simulation`` belongs to one conversation; ``tool_definitions`` are this
    request bound schemas. Keeping accounting here ensures a shared model and
    specialized sample fixtures use identical token and cost semantics.
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
            "usage_basis": f"{TOKEN_ESTIMATE_BASIS}; simulated immediate cache reuse",
        }
    )
    return response
