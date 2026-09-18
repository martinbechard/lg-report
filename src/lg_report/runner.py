"""DeepAgents entry points and a reusable synchronous invocation recorder."""

import json
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langsmith import tracing_context
from pydantic import PrivateAttr

from .cache_policy import CACHE_TTL
from .capture import TraceCapture
from .demo_meter import ContextSimulation, message_record
from .normalize import normalize
from .pricing import Prices
from .render import render
from .schema import Run


class ScriptedChatModel(FakeMessagesListChatModel):
    """Offline demo model; executes the real agent graph without a provider request."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _get_ls_params(self, stop=None, **kwargs):
        return {
            "ls_provider": "demo",
            "ls_model_name": "scripted-chat",
            "ls_model_type": "chat",
        }


class MeteredDemoModel(ScriptedChatModel):
    """Simulate an append-only context and reuse the previous request's prefix."""

    _simulation: ContextSimulation = PrivateAttr(default_factory=ContextSimulation)
    _tools: list[dict] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self._tools = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(self, messages, *args, **kwargs):
        result = super()._generate(messages, *args, **kwargs)
        response = result.generations[0].message.model_copy(deep=True)
        entry, roles = self._simulation.record(
            self._tools, [message_record(m) for m in messages], message_record(response)
        )
        reasoning = response.response_metadata.get("simulated_reasoning_tokens", 0)
        response.usage_metadata = {
            "input_tokens": entry["request_tokens"],
            "output_tokens": entry["response_tokens"] + reasoning,
            "total_tokens": entry["request_tokens"]
            + entry["response_tokens"]
            + reasoning,
            "output_token_details": {"reasoning": reasoning},
            "input_token_details": {"cache_read": entry["cache_read_tokens"]},
        }
        response.response_metadata.update(
            {
                "demo_meter": {
                    **{
                        f"simulated_{role}_tokens": count
                        for role, count in roles.items()
                    },
                    "context_ledger": json.dumps(entry),
                },
                "usage_basis": "Simulated canonical-JSON words/punctuation; completed conversation is cached",
            }
        )
        result.generations[0].message = response
        return result


def record_run(
    agent,
    inputs,
    directory: Path,
    prices: Prices,
    *,
    provider: str,
    model: str,
    title="Chat agent run",
    demo=False,
    include_output=False,
    config=None,
):
    """Save raw trace, normalized JSON, pricing snapshot, and HTML even on agent errors.

    Pass a fresh directory for each invocation. Existing artifacts are never overwritten.
    Other LangGraph applications can reuse this function with their own compiled graph.
    """
    directory.mkdir(parents=True, exist_ok=False)
    capture = TraceCapture(
        directory / "spans.jsonl", provider, model, capture_content=include_output
    )
    result = None
    status = "ok"
    try:
        run_config = {"recursion_limit": 30, **(config or {})}
        run_config["metadata"] = {
            **run_config.get("metadata", {}),
            "cache_ttl": CACHE_TTL,
        }
        run_config["callbacks"] = [*(run_config.get("callbacks") or []), capture]
        # This command owns local capture. Do not inherit ambient hosted tracing settings.
        with tracing_context(enabled=False):
            result = agent.invoke(inputs, config=run_config)
        if isinstance(result, dict) and result.get("__interrupt__"):
            status = "interrupted"
        return result
    except BaseException:
        status = "error"
        raise
    finally:
        capture.close()
        output = None
        if include_output and isinstance(result, dict) and result.get("messages"):
            output = str(result["messages"][-1].content)
        run = (
            normalize(
                directory / "spans.jsonl",
                title=title,
                demo=demo,
                status=status,
                output=output,
            )
            if (directory / "spans.jsonl").stat().st_size
            else Run(
                id=str(uuid4()),
                title=title,
                demo=demo,
                status="error" if status == "error" else "incomplete",
                steps=[],
            )
        )
        (directory / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / "prices.json").write_text(
            prices.model_dump_json(indent=2), encoding="utf-8"
        )
        render(run, prices, directory / "report.html")


class ConversationAgent:
    """Run consecutive user turns, carrying message history between invocations."""

    def __init__(self, agent, requests):
        self.agent = agent
        self.requests = requests

    def invoke(self, inputs, config):
        history = []
        result = None
        for turn, request in enumerate(self.requests, 1):
            turn_config = {
                **config,
                "metadata": {**config.get("metadata", {}), "report_turn": turn},
            }
            result = self.agent.invoke(
                {"messages": [*history, {"role": "user", "content": request}]},
                config=turn_config,
            )
            history = result["messages"]
            if result.get("__interrupt__"):
                break
        return result
