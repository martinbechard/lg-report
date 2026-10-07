"""Exercise shared gateway decisions through both real adapter entry points.

Only the CLI transport is replaced. LangChain binding, graph tool execution,
history translation, and usage preservation run normally without paid requests.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent_runtime.harness.codex_model import CodexChatModel
from agent_runtime.harness.copilot_model import CopilotChatModel
from agent_runtime.harness.gateway_model import gateway_response_schema, text_result


@tool
def read_note() -> str:
    """Read a local observation through the graph, never the gateway CLI."""
    return "local observation"


@pytest.fixture(params=[CodexChatModel, CopilotChatModel])
def gateway(request, monkeypatch):
    """Replace transport I/O while retaining each adapter's result translation."""
    model = request.param(model_name="test-model")
    pending, requests = [], []

    def respond(messages, stop, kwargs):
        """Capture serialized history and return provider usage with model text."""
        requests.append(model.prepare_history(messages, stop, kwargs))
        return text_result(pending.pop(0), {"provider": model.provider},
                           usage={"input_tokens": 100, "output_tokens": 20})

    if isinstance(model, CodexChatModel):
        # Both CLI paths share the parser after process cleanup. A fake process
        # provides the same JSON wire events without launching any executable.
        from contextlib import contextmanager

        @contextmanager
        def command(self, messages, stop, kwargs):
            result = respond(messages, stop, kwargs)
            wire = "\n".join(json.dumps(event) for event in [
                {"type": "item.completed", "item": {"type": "agent_message", "text": result.generations[0].message.content}},
                {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
            ]).encode()
            process = SimpleNamespace(returncode=0, communicate=lambda *a, **k: (wire, b""))

            async def communicate(*args, **kwargs):
                """Emulate nonblocking CLI stdout with the same wire receipt."""
                return wire, b""

            async def start(*args, **kwargs):
                """Construct the async process without invoking a local CLI."""
                return SimpleNamespace(returncode=0, communicate=communicate)

            with patch("agent_runtime.harness.codex_model.subprocess.Popen", return_value=process), patch(
                "agent_runtime.harness.codex_model.asyncio.create_subprocess_exec", side_effect=start
            ):
                yield ["fake"], b""

        monkeypatch.setattr(CodexChatModel, "request", command)
    else:
        def submit(self, messages, stop, kwargs):
            """Return the same completed future as the SDK event-loop boundary."""
            future = Future()
            future.set_result(respond(messages, stop, kwargs))
            return future

        monkeypatch.setattr(CopilotChatModel, "_request", submit)
    return model, pending, requests


@pytest.mark.parametrize("asynchronous", [False, True])
def test_graph_executes_tools_and_returns_observation(gateway, asynchronous):
    """A complete graph loop proves calls and observations cross both adapters."""
    model, pending, requests = gateway
    pending.extend([json.dumps({"content": "", "tool_calls": [{"name": "read_note", "args": {}}]}),
                    json.dumps({"content": "Read successfully", "tool_calls": []})])
    graph = create_agent(model, tools=[read_note])
    payload = {"messages": [HumanMessage("Read the note")]}
    result = asyncio.run(graph.ainvoke(payload)) if asynchronous else graph.invoke(payload)
    assert result["messages"][-1].content == "Read successfully"
    observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert observation.content == "local observation"
    history = json.loads(requests[1][1])
    assert history[-1]["tool_call_id"] == history[-2]["tool_calls"][0]["id"]
    assert history[-1]["content"] == "local observation"
    assert json.loads(requests[0][0].splitlines()[-1])["tools"][0]["name"] == "read_note"
    assert all(m.usage_metadata["total_tokens"] == 120 for m in result["messages"] if isinstance(m, AIMessage))


@pytest.mark.parametrize("response,choice,parallel", [
    ("not JSON", "auto", True),
    ('{"content":"secret"}', "auto", True),
    ('{"content":"","tool_calls":[{"name":"unbound","args":{}}]}', "auto", True),
    ('{"content":"","tool_calls":[{"name":"read_note","args":"bad"}]}', "auto", True),
    ('{"content":"done","tool_calls":[]}', "required", True),
    ('{"content":"done","tool_calls":[]}', "read_note", True),
    ('{"content":"","tool_calls":[{"name":"read_note","args":{}}]}', "none", True),
    ('{"content":"","tool_calls":[{"name":"read_note","args":{}},{"name":"read_note","args":{}}]}', "auto", False),
])
def test_invalid_decisions_fail_without_executing_tools(gateway, response, choice, parallel):
    """Malformed output and violated caller controls never become tool actions."""
    model, pending, _ = gateway
    pending.append(response)
    bound = model.bind_tools([read_note], tool_choice=choice, parallel_tool_calls=parallel)
    with pytest.raises(ValueError, match="Gateway"):
        bound.invoke("Read the note")


def test_bindings_are_independent_and_plain_calls_stay_plain(gateway):
    """A role's bound tools cannot leak into a summary or adaptive user request."""
    model, pending, requests = gateway
    bound = model.bind_tools([read_note], tool_choice={"type": "function", "function": {"name": "read_note"}})
    pending.extend(['{"content":"","tool_calls":[{"name":"read_note","args":{}}]}',
                    '{"done":true,"reason":"goal met","message":""}'])
    assert bound.invoke("read").tool_calls[0]["name"] == "read_note"
    assert json.loads(model.invoke("assess").content)["done"] is True
    assert '"tools"' not in requests[-1][0]


def test_generation_text_matches_decoded_content(gateway):
    """Callbacks and batch consumers must not receive the raw JSON envelope."""
    model, pending, _ = gateway
    pending.append('{"content":"finished","tool_calls":[]}')
    schema = model.bind_tools([read_note]).kwargs
    result = model.generate([[HumanMessage("answer")]], **schema)
    generation = result.generations[0][0]
    assert generation.text == generation.message.content == "finished"


def test_failed_tool_observation_survives_summary_history(gateway):
    """A summary receives errors and call IDs even with no tools bound itself."""
    model, pending, requests = gateway
    pending.append("The read failed; no result was obtained.")
    model.invoke([
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "read_note", "args": {}}]),
        ToolMessage(content="Read denied", tool_call_id="call_1", name="read_note", status="error"),
        HumanMessage("Summarize the history"),
    ])
    history = json.loads(requests[0][1])
    assert history[1] == {"role": "tool", "tool_call_id": "call_1", "name": "read_note",
                          "content": "Read denied", "status": "error"}


def test_response_schema_preserves_typed_arguments_without_mutating_tools():
    """Structured output must constrain argument values, not just a JSON string."""
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"},
                                           "limit": {"type": "integer", "default": 20}},
        "required": ["path"],
    }}}]
    original = json.dumps(tools)
    schema = gateway_response_schema(tools)
    args = schema["properties"]["tool_calls"]["items"]["anyOf"][0]["properties"]["args"]
    assert args["additionalProperties"] is False
    assert args["required"] == ["path", "limit"]
    assert args["properties"]["limit"] == {"type": "integer", "default": 20}
    assert json.dumps(tools) == original


@pytest.mark.parametrize("argument_schema", [
    {"type": "object"},
    {"type": "object", "additionalProperties": {"type": "string"}},
    {"$ref": "#/$defs/Arguments"},
])
def test_open_schemas_keep_the_prompted_protocol(argument_schema):
    """A CLI constraint must not narrow tools that accept arbitrary JSON keys."""
    assert gateway_response_schema([{"function": {
        "name": "store_json", "parameters": argument_schema,
    }}]) is None
