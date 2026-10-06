"""Verify provider substitution and shared accounting without paid requests.

A provider outside the built-in set must work through selection, live-mode
inference, construction, and shutdown. Local text restrictions must not narrow
native API functionality, and missing usage must never become a zero estimate.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_runtime.harness.argument_parser import resolve_live_mode
from agent_runtime.harness.codex_model import CodexChatModel
from agent_runtime.harness.copilot_model import CopilotChatModel
from agent_runtime.harness.model_config import configured_identity, configured_model
from agent_runtime.harness.model_providers import (
    PROVIDERS,
    ModelProvider,
    ProviderPolicy,
    close_model_providers,
)
from agent_runtime.harness.text_model import text_result


@pytest.fixture(autouse=True)
def isolated_environment():
    """Prevent workstation credentials or selected models affecting assertions."""
    with patch.dict(os.environ, {}, clear=True):
        yield


def test_independent_provider_works_without_factory_branches(monkeypatch):
    """Exercise structural conformance rather than subclassing the built-ins."""
    built = FakeListChatModel(responses=["ok"])
    calls = []

    class ExampleProvider:
        """An independent implementation with its own model construction."""

        policy = ProviderPolicy("example", "example-default", "EXAMPLE_KEY", False, True)

        def create_model(self, model, settings):
            """Record exact configuration received across the protocol boundary."""
            calls.append((model, settings["EXAMPLE_KEY"]))
            return built

        def close(self):
            """Record that the same provider participates in common shutdown."""
            calls.append("closed")

    provider: ModelProvider = ExampleProvider()
    monkeypatch.setitem(PROVIDERS, "example", provider)
    settings = {"LG_PROVIDER": "example", "EXAMPLE_KEY": "placeholder"}
    assert configured_identity(settings=settings) == ("example", "example-default")
    assert resolve_live_mode(SimpleNamespace(demo=False, live=None), settings)
    assert not resolve_live_mode(SimpleNamespace(demo=False, live=None), {"LG_PROVIDER": "example"})
    adapter, name, model = configured_model(settings=settings)
    assert adapter is built and (name, model) == ("example", "example-default")
    assert calls == [("example-default", "placeholder")]
    close_model_providers()
    assert calls[-1] == "closed"


@pytest.mark.parametrize("name,credential,class_module,class_name", [
    ("openai", "OPENAI_API_KEY", "langchain_openai", "ChatOpenAI"),
    ("anthropic", "ANTHROPIC_API_KEY", "langchain_anthropic", "ChatAnthropic"),
])
def test_api_models_retain_native_tool_binding(name, credential, class_module, class_name):
    """A protocol must not replace the API adapter with a text-only wrapper."""
    from importlib import import_module

    adapter, _, _ = configured_model(settings={"LG_PROVIDER": name, credential: "placeholder"})
    assert isinstance(adapter, getattr(import_module(class_module), class_name))
    tool = {"name": "lookup", "description": "Find a value", "parameters": {"type": "object", "properties": {}}}
    bound = adapter.bind_tools([tool], tool_choice="lookup")
    assert bound is not adapter and bound.kwargs["tools"]


@pytest.mark.parametrize("model_type", [CodexChatModel, CopilotChatModel])
def test_local_history_and_trace_contract(model_type):
    """Both local transports receive the same roles and reject lossy coercions."""
    model = model_type(model_name="chosen", reasoning_effort="low")
    instructions, history = model.prepare_history([
        SystemMessage("Follow these instructions"), HumanMessage("Hi"), AIMessage("Hello"),
    ], None, {})
    assert instructions.startswith("Follow these instructions")
    assert history == '[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]'
    assert model._identifying_params["reasoning_effort"] == "low"
    assert model._get_ls_params()["ls_provider"] == model.provider
    with pytest.raises(ValueError, match="tool messages"):
        model.prepare_history([ToolMessage("result", tool_call_id="1")], None, {})
    with pytest.raises(TypeError, match="text messages"):
        model.prepare_history([HumanMessage(content=[{"type": "text", "text": "Hi"}])], None, {})


@pytest.mark.parametrize("provider", ["codex", "copilot"])
def test_inclusive_receipts_and_unknown_usage(provider):
    """Cache and reasoning partition totals; partial receipts remain auditable."""
    receipt = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 40,
               "cache_creation_input_tokens": 10, "reasoning_output_tokens": 5}
    metadata = {"provider": provider, "model_name": "chosen", "usage": receipt}
    message = text_result("answer", metadata).generations[0].message
    assert message.usage_metadata == {
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_token_details": {"cache_read": 40, "cache_creation": 10},
        "output_token_details": {"reasoning": 5},
    }
    metadata["usage"] = {"input_tokens": 100}
    message = text_result("answer", metadata).generations[0].message
    assert message.usage_metadata is None
    assert message.response_metadata["usage"] == {"input_tokens": 100}
    metadata["usage"] = {**receipt, "cache_read_input_tokens": 101}
    with pytest.raises(ValueError, match="cache usage"):
        text_result("answer", metadata)


def test_shutdown_attempts_remaining_providers_after_error(monkeypatch):
    """One broken transport cannot strand another provider's shared process."""
    closed = []

    def close_bad():
        """Simulate a failed shutdown after its callback was reached."""
        closed.append("bad")
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr("agent_runtime.harness.model_providers.PROVIDERS", {
        "good": SimpleNamespace(close=lambda: closed.append("good")),
        "bad": SimpleNamespace(close=close_bad),
    })
    with pytest.raises(RuntimeError, match="shutdown failed"):
        close_model_providers()
    assert closed == ["bad", "good"]


def test_terminal_construction_failure_closes_registered_providers(monkeypatch):
    """A failed graph build still releases runtimes through the common boundary."""
    from unittest.mock import Mock

    from agent_runtime.harness.execute_conversation import execute_conversation

    close = Mock()
    monkeypatch.setattr("agent_runtime.harness.model_providers.PROVIDERS", {
        "example": SimpleNamespace(close=close),
    })
    catalog = SimpleNamespace(
        get=lambda _: SimpleNamespace(tracing="local"),
        create_run=Mock(side_effect=ValueError("bad graph")),
    )
    with pytest.raises(ValueError, match="bad graph"):
        execute_conversation(catalog=catalog, sample_id="sample",
                             settings=SimpleNamespace(live=True), options={}, client=None)
    close.assert_called_once()


def test_web_shutdown_closes_registered_providers(monkeypatch, tmp_path):
    """The web lifespan uses the same cleanup contract as terminal execution."""
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from agent_runtime.web.server import create_app

    close = Mock()
    monkeypatch.setattr("agent_runtime.harness.model_providers.PROVIDERS", {
        "example": SimpleNamespace(close=close),
    })
    with TestClient(create_app(directory=tmp_path), base_url="http://localhost"):
        close.assert_not_called()
    close.assert_called_once()
