"""Verify the Codex adapter without consuming account quota or invoking tools.

A tiny executable emits the real CLI event shape, exercising subprocess input,
usage capture, cleanup, and cancellation as well as LangChain's sync/async paths.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_runtime.harness.argument_parser import resolve_live_mode
from agent_runtime.harness.codex_model import CodexChatModel, parse_result
from agent_runtime.harness.model_config import configured_identity, configured_model
from agent_runtime.harness.trace_capture import TraceCapture
from reporting.normalize import normalize


@pytest.fixture(autouse=True)
def isolated_environment():
    """Keep developer credentials, provider selection, and CLI settings out of tests."""
    with patch.dict(os.environ, {}, clear=True):
        yield


def events(usage=None):
    """Describe a successful CLI turn, including an informational startup warning."""
    return [
        {"type": "thread.started", "thread_id": "test"},
        {"type": "item.completed", "item": {"type": "error", "message": "Startup warning"}},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "An answer"}},
        {"type": "turn.completed", "usage": usage},
    ]


def jsonl(items):
    """Serialize protocol receipts rather than mocking the adapter's output."""
    return "\n".join(json.dumps(item) for item in items)


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Expose a subprocess with no model/network access and observable lifecycle."""
    if os.name != "posix":
        pytest.skip("Executable fixture uses a POSIX shebang")
    executable = tmp_path / "fake-codex"
    capture = tmp_path / "capture.json"
    receipt = tmp_path / "events.jsonl"
    receipt.write_text(jsonl(events({"input_tokens": 100, "cached_input_tokens": 40,
                                     "output_tokens": 20, "reasoning_output_tokens": 5})))
    executable.write_text(f'#!{sys.executable}\n' + '''"""Emulate Codex receipts for adapter tests without model access.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
directory = Path(args[args.index("--cd") + 1])
Path(os.environ["CODEX_TEST_CAPTURE"]).write_text(json.dumps({
    "args": args, "prompt": json.loads(sys.stdin.read()), "pid": os.getpid(),
    "instructions": (directory / "instructions.txt").read_text(),
}))
if os.environ.get("CODEX_TEST_SLEEP"):
    time.sleep(30)
print(Path(os.environ["CODEX_TEST_EVENTS"]).read_text(), flush=True)
''')
    executable.chmod(0o700)
    monkeypatch.setenv("CODEX_TEST_CAPTURE", str(capture))
    monkeypatch.setenv("CODEX_TEST_EVENTS", str(receipt))
    return CodexChatModel(model_name="chosen", executable=str(executable)), capture


@pytest.mark.parametrize("model", [None, "", "auto"])
def test_explicit_model_required(model):
    """Codex must never select a default model silently."""
    with pytest.raises(ValueError, match="explicit"):
        configured_identity(settings={"LG_PROVIDER": "codex", "LG_MODEL": model})


def test_factory_live_selection_and_restrictions():
    """Construction remains lazy and login selection needs no vendor API key."""
    settings = {"LG_PROVIDER": "codex", "LG_MODEL": "chosen", "LG_CODEX_CLI": "custom-codex",
                "LG_EFFORT": "low"}
    model, provider, code = configured_model(settings=settings)
    assert isinstance(model, CodexChatModel)
    assert (provider, code, model.executable, model.reasoning_effort) == (
        "codex", "chosen", "custom-codex", "low"
    )
    assert resolve_live_mode(SimpleNamespace(demo=False, live=None), settings)
    assert not resolve_live_mode(SimpleNamespace(demo=True, live=None), settings)
    with pytest.raises(ValueError, match="LG_AVAILABLE_MODELS"):
        configured_identity("other", settings={**settings, "LG_AVAILABLE_MODELS": "chosen"})
    with pytest.raises(ValueError, match="LG_MAX_TOKENS"):
        configured_model(settings={**settings, "LG_MAX_TOKENS": "100"})


def test_sync_process_captures_usage_and_isolates_history(fake_cli):
    """Verify actual process arguments, stdin, temp cleanup, and inclusive totals."""
    model, capture = fake_cli
    model.reasoning_effort = "low"
    answer = model.invoke([SystemMessage("Be concise"), HumanMessage("First"),
                           AIMessage("Previous"), HumanMessage("Second")])
    data = json.loads(capture.read_text())
    assert data["prompt"] == [{"role": "user", "content": "First"},
                              {"role": "assistant", "content": "Previous"},
                              {"role": "user", "content": "Second"}]
    assert "Be concise" in data["instructions"]
    assert "--ignore-user-config" in data["args"] and "--ephemeral" in data["args"]
    assert "features.shell_tool=false" in data["args"]
    assert data["args"][data["args"].index("--sandbox") + 1] == "read-only"
    assert not Path(data["args"][data["args"].index("--cd") + 1]).exists()
    assert answer.content == "An answer"
    assert answer.usage_metadata["total_tokens"] == 120
    assert answer.usage_metadata["input_token_details"]["cache_read"] == 40
    assert answer.usage_metadata["output_token_details"]["reasoning"] == 5


def test_async_process_and_report_capture(fake_cli, tmp_path):
    """Codex identity and usage must reach normalized run.json through callbacks."""
    model, _ = fake_cli
    model.reasoning_effort = "low"
    path = tmp_path / "spans.jsonl"
    capture = TraceCapture(path, provider="codex", model="chosen", capture_content=True)
    try:
        asyncio.run(model.ainvoke([HumanMessage("Question")], config={"callbacks": [capture]}))
    finally:
        capture.close()
    run = normalize(path, title="Codex adapter test")
    step = next(s for s in run.steps if s.kind == "model")
    assert (step.provider, step.model, step.status) == ("codex", "chosen", "ok")
    assert step.effort == "low"
    assert step.usage.input_tokens == 100
    assert step.usage.cache_read == 40 and step.usage.reasoning == 5
    assert "An answer" in str(step.response)


@pytest.mark.parametrize("receipt", [None, {}, {"input_tokens": 100}])
def test_missing_usage_is_unknown(receipt):
    """Partial receipts cannot become complete zero-cost usage records."""
    assert parse_result(jsonl(events(receipt)), 0, "chosen").generations[0].message.usage_metadata is None


def test_protocol_failures_do_not_become_answers():
    """A printed answer alone cannot hide failure, tools, or missing completion."""
    with pytest.raises(RuntimeError, match="exit 1"):
        parse_result("secret stderr not included", 1, "chosen")
    with pytest.raises(RuntimeError, match="completed turn"):
        parse_result(jsonl(events()[:-1]), 0, "chosen")
    with pytest.raises(RuntimeError, match="failed"):
        parse_result(jsonl(events() + [{"type": "turn.failed"}]), 0, "chosen")
    with pytest.raises(RuntimeError, match="non-text"):
        parse_result(jsonl(events() + [{"type": "item.completed", "item": {
            "type": "command_execution"}}]), 0, "chosen")
    with pytest.raises(ValueError):
        parse_result("not JSON", 0, "chosen")
    with pytest.raises(ValueError, match="cache usage"):
        parse_result(jsonl(events({"input_tokens": 1, "output_tokens": 0,
                                  "cached_input_tokens": 2})), 0, "chosen")


def test_unsupported_inputs_fail_before_process_creation(fake_cli):
    """Do not silently reinterpret tool results, images, forced tools, or stops."""
    model, capture = fake_cli
    for messages in ([ToolMessage(content="tool", tool_call_id="1")],
                     [AIMessage(content="", tool_calls=[{"id": "1", "name": "x", "args": {}}])]):
        with pytest.raises(ValueError, match="tool messages"):
            model.invoke(messages)
    with pytest.raises(TypeError, match="text messages"):
        model.invoke([HumanMessage(content=[{"type": "text", "text": "image"}])])
    with pytest.raises(ValueError, match="stop"):
        model.invoke([HumanMessage("Hi")], stop=["stop"])
    with pytest.raises(ValueError, match="graph tools"):
        model.bind_tools([], tool_choice="required")
    assert model.bind_tools([]) is model
    assert not capture.exists()


def test_timeout_reaps_process(fake_cli, monkeypatch):
    """An expired request must stop its subprocess before deleting instructions."""
    model, capture = fake_cli
    model.request_timeout = 0.5
    monkeypatch.setenv("CODEX_TEST_SLEEP", "1")
    with pytest.raises(TimeoutError):
        model.invoke([HumanMessage("Hi")])
    data = json.loads(capture.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(data["pid"], 0)
    assert not Path(data["args"][data["args"].index("--cd") + 1]).exists()


def test_async_cancellation_reaps_process(fake_cli, monkeypatch):
    """Web request cancellation must not leave paid generation running."""
    model, capture = fake_cli
    monkeypatch.setenv("CODEX_TEST_SLEEP", "1")

    async def cancel():
        """Cancel after the fake process confirms startup, then await cleanup."""
        task = asyncio.create_task(model.ainvoke([HumanMessage("Hi")]))
        for _ in range(100):
            if capture.exists():
                break
            await asyncio.sleep(0.02)
        assert capture.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    with pytest.raises(ProcessLookupError):
        os.kill(json.loads(capture.read_text())["pid"], 0)
