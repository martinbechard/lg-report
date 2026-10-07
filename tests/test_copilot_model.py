"""Check Copilot lifecycle and tool-free model requests without paid generation.

Fake SDK endpoints prove handshake-based reuse and occupied-port handling. Real
LangChain invocation exercises callbacks and async bridging without credentials.
AI attribution: Generated with AI assistance by Avery Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_runtime.harness import copilot_model
from agent_runtime.harness.argument_parser import resolve_live_mode
from agent_runtime.harness.model_config import configured_identity, configured_model


@pytest.fixture
def isolated(monkeypatch):
    """Never let a developer's provider choices or SDK sockets leak into tests."""
    with patch.dict(os.environ, {}, clear=True):
        yield
    copilot_model.close_copilot_servers()


@pytest.mark.parametrize("model", [None, "", "auto"])
def test_explicit_model_required(isolated, model):
    """Copilot must never choose a model implicitly or through auto routing."""
    with pytest.raises(ValueError, match="explicit"):
        configured_identity(settings={"LG_PROVIDER": "copilot", "LG_MODEL": model})


def test_factory_is_lazy_and_live_without_vendor_keys(isolated):
    """Construction performs no network work and demo remains an explicit opt-out."""
    pytest.importorskip("copilot")
    settings = {"LG_PROVIDER": "copilot", "LG_MODEL": "gpt-5.4"}
    with patch.object(copilot_model, "server_for") as server:
        model, provider, code = configured_model(settings=settings)
        assert isinstance(model, copilot_model.CopilotChatModel)
        assert (provider, code) == ("copilot", "gpt-5.4")
        server.assert_not_called()
    assert resolve_live_mode(SimpleNamespace(static=False, live=None), settings)
    assert not resolve_live_mode(SimpleNamespace(static=True, live=None), settings)
    with pytest.raises(ValueError, match="LG_MAX_TOKENS"):
        configured_model(settings={**settings, "LG_MAX_TOKENS": "100"})
    with pytest.raises(ValueError, match="LG_AVAILABLE_MODELS"):
        configured_identity("other", settings={**settings, "LG_AVAILABLE_MODELS": "gpt-5.4"})


@pytest.fixture
def runtime(tmp_path):
    """Run coroutines directly while replacing only the real SDK boundary."""
    server = copilot_model.CopilotServer(str(tmp_path / "copilot.json"))
    yield server
    server.close()


def test_first_free_port_and_recorded_reuse(runtime):
    """Skip a foreign service, publish 7002, then reuse the healthy connection."""
    pytest.importorskip("copilot")
    failed = SimpleNamespace(start=AsyncMock(side_effect=TimeoutError), stop=AsyncMock())
    good = SimpleNamespace(start=AsyncMock(), ping=AsyncMock(), stop=AsyncMock())
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.bind.side_effect = [OSError("occupied"), None]
    with patch.object(copilot_model.socket, "socket", return_value=probe), patch(
        "copilot.CopilotClient", side_effect=[failed, good]
    ) as factory:
        # Use a separate runner: socket mocking must not affect event-loop setup.
        assert runtime.submit(runtime.connect()).result() is good
        assert runtime.submit(runtime.connect()).result() is good
    saved = json.loads(runtime.path.read_text())
    assert saved["port"] == 7002
    assert saved["connection_token"]
    assert runtime.path.stat().st_mode & 0o777 == 0o600
    assert factory.call_count == 2
    assert factory.call_args.kwargs["mode"] == "copilot-cli"
    assert factory.call_args.kwargs["use_logged_in_user"] is True
    failed.stop.assert_awaited_once()


def test_stale_port_restarts_at_7001(runtime):
    """A saved high port does not reserve it when lower ports are now free."""
    pytest.importorskip("copilot")
    runtime.path.write_text('{"port": 7010}')
    good = SimpleNamespace(start=AsyncMock(), ping=AsyncMock(), stop=AsyncMock())
    probe = MagicMock()
    probe.__enter__.return_value = probe
    with patch.object(copilot_model.socket, "socket", return_value=probe), patch(
        "copilot.CopilotClient", return_value=good
    ):
        runtime.submit(runtime.connect()).result()
    assert json.loads(runtime.path.read_text())["port"] == 7001


def test_saved_server_handshake_reuse(runtime):
    """A saved live server is connected to, not replaced by an owned child."""
    pytest.importorskip("copilot")
    runtime.path.write_text('{"port": 7010, "connection_token": "local-secret"}')
    good = SimpleNamespace(start=AsyncMock(), ping=AsyncMock(), stop=AsyncMock())
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.bind.side_effect = OSError("occupied")
    with patch.object(copilot_model.socket, "socket", return_value=probe), patch(
        "copilot.CopilotClient", return_value=good
    ) as factory:
        runtime.submit(runtime.connect()).result()
    connection = factory.call_args.kwargs["connection"]
    assert connection.url == "127.0.0.1:7010"
    assert connection.connection_token == "local-secret"


def fake_client(usage=None, error=None):
    """Emulate session events and release calls, retaining arguments for assertions."""
    listeners = []
    session = SimpleNamespace(
        session_id="temporary", disconnect=AsyncMock(),
        on=MagicMock(side_effect=lambda listener: listeners.append(listener) or MagicMock()),
    )

    async def send(prompt, **kwargs):
        """Deliver the usage receipt before idle, as the SDK's request does."""
        if error:
            raise error
        if usage:
            listeners[0](SimpleNamespace(type=SimpleNamespace(value="assistant.usage"), data=usage))
        return SimpleNamespace(data=SimpleNamespace(content="answer"))

    session.send_and_wait = AsyncMock(side_effect=send)
    client = SimpleNamespace(
        get_auth_status=AsyncMock(return_value=SimpleNamespace(isAuthenticated=True)),
        list_models=AsyncMock(return_value=[SimpleNamespace(id="gpt-5.4")]),
        create_session=AsyncMock(return_value=session), delete_session=AsyncMock(),
    )
    return client, session


@pytest.mark.parametrize("receipt", [None, (12, 4), (12, None)])
def test_request_disables_tools_and_preserves_unknown_usage(runtime, receipt):
    """Session options enforce tool-free behavior; absent counts are never zero."""
    pytest.importorskip("copilot")
    usage = SimpleNamespace(model="gpt-5.4", input_tokens=receipt[0], output_tokens=receipt[1]) if receipt else None
    client, session = fake_client(usage)
    with patch.object(runtime, "connect", AsyncMock(return_value=client)):
        response = runtime.submit(runtime.request("gpt-5.4", "system", "prompt", None, 10)).result()
    options = client.create_session.call_args.kwargs
    assert options["available_tools"] == options["tools"] == []
    assert options["system_message"] == {"mode": "replace", "content": "system"}
    assert options["infinite_sessions"] == {"enabled": False}
    assert options["enable_config_discovery"] is False
    assert options["enable_skills"] is False
    assert options["enable_file_hooks"] is False
    assert options["skip_custom_instructions"] is True
    assert options["on_permission_request"](None, None).kind == "reject"
    message = response.generations[0].message
    assert message.content == "answer"
    if receipt == (12, 4):
        assert message.usage_metadata == {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
    else:
        assert message.usage_metadata is None
    session.disconnect.assert_awaited_once()
    client.delete_session.assert_awaited_once_with("temporary")


def test_failed_request_cleans_session(runtime):
    """A timed-out generation remains a failure and releases temporary history."""
    pytest.importorskip("copilot")
    client, session = fake_client(error=TimeoutError("generation"))
    with patch.object(runtime, "connect", AsyncMock(return_value=client)), pytest.raises(TimeoutError):
        runtime.submit(runtime.request("gpt-5.4", "s", "p", None, 10)).result()
    session.disconnect.assert_awaited_once()
    client.delete_session.assert_awaited_once()


def test_unavailable_model_never_creates_session(runtime):
    """The account catalog is checked before sending a billable request."""
    pytest.importorskip("copilot")
    client, _ = fake_client()
    with patch.object(runtime, "connect", AsyncMock(return_value=client)), pytest.raises(ValueError, match="unavailable"):
        runtime.submit(runtime.request("wrong-model", "s", "p", None, 10)).result()
    client.create_session.assert_not_awaited()


def test_sync_async_history_and_binding(isolated, tmp_path):
    """Plain public invocations preserve role order across synchronous and async calls."""
    model = copilot_model.CopilotChatModel(model_name="gpt-5.4", config_path=str(tmp_path / "c.json"))
    server = copilot_model.server_for(model.config_path)
    result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
    messages = [SystemMessage(content="system"), HumanMessage(content="one"), AIMessage(content="two"), HumanMessage(content="three")]
    with patch.object(server, "request", AsyncMock(return_value=result)) as request:
        assert model.invoke(messages).content == "ok"
        assert asyncio.run(model.ainvoke(messages)).content == "ok"
    assert json.loads(request.call_args.args[2]) == [
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    with pytest.raises(ValueError, match="requires bound tools"):
        model.bind_tools([], tool_choice="required")
    assert json.loads(model.prepare_history([ToolMessage(content="tool", tool_call_id="1")], None, {})[1])[0]["tool_call_id"] == "1"


def test_simple_chat_graph_with_tool_free_adapter(isolated, tmp_path):
    """The simple-chat graph sends a plain request with no gateway tool envelope."""
    from agent_runtime.agents.chat_agent import build_agent

    model = copilot_model.CopilotChatModel(model_name="gpt-5.4", config_path=str(tmp_path / "graph.json"))
    server = copilot_model.server_for(model.config_path)
    result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="Hello"))])
    with patch.object(server, "request", AsyncMock(return_value=result)):
        graph = build_agent({"model": model})
        response = graph.invoke({"messages": [HumanMessage(content="Hello")]})
    assert response["messages"][-1].content == "Hello"


def test_wrong_returned_model_is_not_reported_as_requested(runtime):
    """A runtime substitution must fail rather than mislabel accounting evidence."""
    pytest.importorskip("copilot")
    client, session = fake_client(SimpleNamespace(model="unexpected"))
    with patch.object(runtime, "connect", AsyncMock(return_value=client)), pytest.raises(RuntimeError, match="different model"):
        runtime.submit(runtime.request("gpt-5.4", "s", "p", None, 10)).result()
    session.disconnect.assert_awaited_once()


def test_start_failure_is_not_hidden_or_persisted(runtime):
    """A broken runtime on a free port fails promptly without a false ready file."""
    pytest.importorskip("copilot")
    failed = SimpleNamespace(start=AsyncMock(side_effect=RuntimeError("broken runtime")), stop=AsyncMock())
    probe = MagicMock()
    probe.__enter__.return_value = probe
    with patch.object(copilot_model.socket, "socket", return_value=probe), patch(
        "copilot.CopilotClient", return_value=failed
    ), pytest.raises(RuntimeError, match="broken runtime"):
        runtime.submit(runtime.connect()).result()
    assert not runtime.path.exists()
    failed.stop.assert_awaited_once()
