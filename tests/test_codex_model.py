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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
    "schema": json.loads((directory / "response-schema.json").read_text()) if "--output-schema" in args else None,
    "catalog": json.loads((directory / "model-catalog.json").read_text()) if (directory / "model-catalog.json").exists() else None,
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
                "LG_EFFORT": "low", "LG_CODEX_MODEL_CATALOG": "/not-read-at-construction.json"}
    model, provider, code = configured_model(settings=settings)
    assert isinstance(model, CodexChatModel)
    assert model.model_catalog_file == "/not-read-at-construction.json"
    assert (provider, code, model.executable, model.reasoning_effort) == (
        "codex", "chosen", "custom-codex", "low"
    )
    assert resolve_live_mode(SimpleNamespace(static=False, live=None), settings)
    assert not resolve_live_mode(SimpleNamespace(static=True, live=None), settings)
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
    assert "skills.include_instructions=false" in data["args"]
    assert "features.goals=false" in data["args"]
    assert "agents.enabled=false" in data["args"]
    assert "features.multi_agent_v2=false" in data["args"]
    assert "features.sleep_tool=false" in data["args"]
    assert "tools.experimental_request_user_input.enabled=false" in data["args"]
    assert "include_environment_context=false" in data["args"]
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


def test_tool_decision_uses_cli_schema_and_preserves_usage(fake_cli):
    """The real subprocess boundary receives a schema and returns graph calls."""
    model, capture = fake_cli
    wire = events({"input_tokens": 100, "output_tokens": 20})
    wire[3]["item"]["text"] = json.dumps({"content": "", "tool_calls": [
        {"name": "read_file", "args": {"path": "/plan.md"}},
    ]})
    Path(os.environ["CODEX_TEST_EVENTS"]).write_text(jsonl(wire))
    answer = model.bind_tools([{"name": "read_file", "description": "Read a virtual file",
                               "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]).invoke("Read the plan")
    data = json.loads(capture.read_text())
    assert data["schema"]["additionalProperties"] is False
    assert data["schema"]["properties"]["tool_calls"]["items"]["anyOf"][0]["properties"]["args"]["properties"]["path"]["type"] == "string"
    assert "features.shell_tool=false" in data["args"]
    assert answer.tool_calls[0]["args"] == {"path": "/plan.md"}
    assert answer.usage_metadata["total_tokens"] == 120


def test_open_tool_schema_is_sent_once_in_prompt(fake_cli):
    """Removing duplication must preserve arbitrary-key tools and their defaults."""
    model, _capture = fake_cli
    tools = [{"type": "function", "function": {"name": "store", "description": "Store arbitrary keys",
              "parameters": {"type": "object", "additionalProperties": {"type": "string"}}}}]
    with model.request([HumanMessage("Store the supplied object")], None,
                       {"tools": tools, "tool_choice": "required", "parallel_tool_calls": False}) as (command, _prompt):
        directory = Path(command[command.index('--cd') + 1])
        instructions = (directory / 'instructions.txt').read_text()
        assert '--output-schema' not in command
        assert 'Store arbitrary keys' in instructions
        assert '"additionalProperties":{"type":"string"}' in instructions
        assert '"tool_choice":"required"' in instructions
        assert '"parallel_tool_calls":false' in instructions


def test_catalog_removes_native_tools_without_changing_model_settings(fake_cli, tmp_path):
    """A private metadata copy changes tools only and preserves the account cache."""
    model, capture = fake_cli
    selected = {"slug": "chosen", "tool_mode": "code_mode_only", "shell_type": "unified_exec",
                "apply_patch_tool_type": "freeform", "experimental_supported_tools": ["clock"],
                "use_responses_lite": True, "context_window": 200000,
                "supported_reasoning_levels": [{"effort": "medium", "description": "Medium"}]}
    original = json.dumps({"identity": "not-for-cli", "models": [selected, {"slug": "unrelated"}]})
    source = tmp_path / 'models.json'
    source.write_text(original)
    model.model_catalog_file = str(source)
    wire = events({"input_tokens": 100, "output_tokens": 20})
    wire[3]['item']['text'] = '{"content":"Done","tool_calls":[]}'
    Path(os.environ['CODEX_TEST_EVENTS']).write_text(jsonl(wire))
    model.bind_tools([{"name": "read", "description": "Read the virtual file",
                       "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]).invoke('Answer the question')
    data = json.loads(capture.read_text())
    expected = {**selected, "tool_mode": "direct", "shell_type": "disabled",
                "apply_patch_tool_type": None, "experimental_supported_tools": []}
    assert data['catalog'] == {"models": [expected]}
    assert data['schema'] is None
    assert data['instructions'].count('Read the virtual file') == 1
    assert '"parameters"' in data['instructions']
    assert source.read_text() == original
    assert 'include_permissions_instructions=false' in data['args']
    assert data['args'][data['args'].index('--sandbox') + 1] == 'read-only'
    assert 'approval_policy="never"' in data['args']
    assert not Path(data['args'][data['args'].index('--cd') + 1]).exists()


@pytest.mark.parametrize('entries', [[], [{"slug": "other"}], [{"slug": "chosen"}, {"slug": "chosen"}]])
def test_catalog_requires_exact_model_metadata(fake_cli, tmp_path, entries):
    """Missing/ambiguous entries fail before spawning rather than guessing defaults."""
    model, capture = fake_cli
    source = tmp_path / 'models.json'
    source.write_text(json.dumps({"models": entries}))
    model.model_catalog_file = str(source)
    with pytest.raises(ValueError, match='exactly one entry'):
        model.invoke('Answer the question')
    assert not capture.exists()


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
    """Reject unsupported images, stops, and required calls without schemas."""
    model, capture = fake_cli
    with pytest.raises(TypeError, match="text messages"):
        model.invoke([HumanMessage(content=[{"type": "text", "text": "image"}])])
    with pytest.raises(ValueError, match="stop"):
        model.invoke([HumanMessage("Hi")], stop=["stop"])
    with pytest.raises(ValueError, match="requires bound tools"):
        model.bind_tools([], tool_choice="required")
    assert model.bind_tools([]).bound is model
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
