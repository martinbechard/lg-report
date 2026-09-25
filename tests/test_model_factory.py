"""Verify factory mode selection and conversation routing independently of providers.

These checks protect caller isolation and explicit live failures. The subagent
integration test separately exercises actual delegation, tools, and accounting.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from agent_runtime.harness import model_factory
from agent_runtime.harness.console_client import ConsoleClient
from samples.subagent_chat.sample import CONVERSATION

# Expected values are test projections of the authored conversation.
FINAL_ANSWER = CONVERSATION[-1]["content"]
USER_PROMPTS = [entry["content"] for entry in CONVERSATION if entry["role"] == "client"]


def test_conversation_routing_and_fresh_models():
    """Interleaved child and tool entries never consume the parent's next answer."""
    with model_factory.model_factory_scope(live=False, conversation=CONVERSATION):
        parent = model_factory.build_model("example-model", caller="workflow")
        child = model_factory.build_model("example-model", caller="isolated-subagent")
        second_parent = model_factory.build_model(caller="workflow")
    # Invocation after scope exit proves existing graphs retain their mode.
    request = HumanMessage(content="delegate")
    config = {"metadata": {"lc_agent_name": "delegating_parent"}}
    delegation = parent.invoke([request], config=config)
    assert delegation.tool_calls[0]["name"] == "task"
    assert (
        child.invoke(
            "echo", config={"metadata": {"lc_agent_name": "isolated-subagent"}}
        ).tool_calls[0]["name"]
        == "echo_tool"
    )
    result = ToolMessage(content="Child summary", tool_call_id="delegate-1")
    assert (
        parent.invoke([request, delegation, result], config=config).content
        == FINAL_ANSWER
    )
    assert (
        second_parent.invoke("delegate", config=config).tool_calls[0]["name"] == "task"
    )
    assert model_factory.client_prompts(CONVERSATION) == USER_PROMPTS
    assert parent.conversation == child.conversation == CONVERSATION
    assert parent._positions == {"delegating_parent": 2}
    assert child._positions == {"isolated-subagent": 1}


def test_missing_caller_and_scope_reset(monkeypatch):
    """A misspelled caller fails locally and cannot leave simulation enabled."""
    sentinel = object()
    calls = []

    def configured(name):
        """Capture explicit model selection without creating a provider client."""
        calls.append(name)
        return sentinel, "provider", name

    monkeypatch.setattr(model_factory, "configured_model", configured)
    with (
        pytest.raises(ValueError, match="missing-agent"),
        model_factory.model_factory_scope(live=False, conversation=CONVERSATION),
    ):
        model_factory.build_model(caller="missing-agent").invoke(
            "Start", config={"metadata": {"lc_agent_name": "missing-agent"}}
        )
    assert model_factory.build_model("chosen-model", caller="workflow") is sentinel
    assert calls == ["chosen-model"]


def test_nested_live_failure_never_falls_back(monkeypatch):
    """Live errors propagate; leaving a nested scope restores its caller's mode."""

    def unavailable(name):
        """Represent a provider configuration failure without network access."""
        raise ValueError("Missing provider credential")

    monkeypatch.setenv("LG_PROVIDER", "openai")
    monkeypatch.setattr(model_factory, "configured_model", unavailable)
    with model_factory.model_factory_scope(live=False, conversation=CONVERSATION):
        with (
            pytest.raises(ValueError, match="Missing provider credential"),
            model_factory.model_factory_scope(live=True, conversation=[]),
        ):
            model_factory.build_model(caller="workflow")
        assert model_factory.build_model(caller="workflow").conversation


def test_subagent_live_build_does_not_load_script(monkeypatch):
    """A live app builds both factory models without touching scripted content.

    Stub only the provider constructor: the actual app, workflow, specialist,
    and DeepAgents compilation still run. No paid request is needed to verify
    this construction boundary.
    """
    from langchain_core.messages import AIMessage

    from agent_runtime.harness import sample_catalog
    from agent_runtime.harness.sample_catalog import SampleCatalog
    from agent_runtime.harness.simulated_model import SimulatedModel

    models = []

    def provider_model(name):
        """Return a distinct valid adapter for each real-mode factory request."""
        model = SimulatedModel(
            conversation=[
                {
                    "role": "test-agent",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                    "response_metadata": response.response_metadata,
                }
                for response in [AIMessage(content="unused")]
            ],
            agent_name="test-agent",
        )
        models.append(model)
        return model, "openai", "selected-model"

    monkeypatch.setenv("LG_PROVIDER", "openai")
    monkeypatch.setenv("LG_MODEL", "selected-model")
    monkeypatch.setattr(model_factory, "configured_model", provider_model)
    original_import = sample_catalog.import_module

    def import_workflow_only(name):
        """Metadata imports are allowed; live execution must bypass fixture factories."""
        module = original_import(name)
        if name.endswith(".sample") and hasattr(module, "build_scripted_models"):
            monkeypatch.setattr(
                module,
                "build_scripted_models",
                lambda options: pytest.fail("Live mode called a simulated factory"),
            )
        return module

    monkeypatch.setattr(sample_catalog, "import_module", import_workflow_only)
    graph, provider, model_id = SampleCatalog().create_run("subagent_chat", True)
    assert graph is not None
    assert (provider, model_id) == ("openai", "selected-model")
    assert len(models) == 2 and models[0] is not models[1]


def test_subagent_static_client_comes_from_conversation():
    """Runtime extraction supplies only client turns and creates fresh clients."""
    from agent_runtime.harness.configure_sample_script import (
        configure_sample_script,
    )
    from agent_runtime.harness.sample_catalog import SampleCatalog

    catalog = SampleCatalog()

    first = ConsoleClient()
    second = ConsoleClient()
    configure_sample_script(first, catalog, "subagent_chat")
    configure_sample_script(second, catalog, "subagent_chat_langfuse")
    assert [first.receive().prompt] == USER_PROMPTS
    assert [second.receive().prompt] == USER_PROMPTS


def test_chronological_extraction_preserves_metadata_and_isolates_runs():
    """Keep reasoning evidence and tool IDs while excluding interruption answers."""
    conversation = [
        {"role": "client", "content": "Begin"},
        {
            "role": "test-agent",
            "content": "Check",
            "tool_calls": [
                {"name": "inspect", "args": {"section": "first"}, "id": "inspect-1"}
            ],
            "response_metadata": {"simulated_reasoning_tokens": 12000},
        },
        {"role": "tool", "content": "Expected observation"},
        {"role": "human", "interaction": "clarification", "content": "Continue"},
        {"role": "test-agent", "content": "Done"},
        {"role": "client", "content": "Next request"},
    ]
    first = model_factory.model_responses(conversation, "test-agent")
    second = model_factory.model_responses(conversation, "test-agent")
    assert [message.content for message in first] == ["Check", "Done"]
    assert model_factory.client_prompts(conversation) == ["Begin", "Next request"]
    assert first[0].response_metadata["simulated_reasoning_tokens"] == 12000
    first[0].tool_calls[0]["args"]["section"] = "changed"
    first[0].response_metadata["simulated_reasoning_tokens"] = 0
    assert second[0].tool_calls[0]["args"]["section"] == "first"
    assert conversation[1]["response_metadata"]["simulated_reasoning_tokens"] == 12000


def test_catalog_meters_rebuilt_quote_requests_without_cache_reuse(monkeypatch):
    """Quote simulation estimates each request without assuming retained transcripts."""
    from agent_runtime.harness.sample_catalog import SampleCatalog
    from agent_runtime.harness.simulated_model import (
        SimulatedModel,
    )

    catalog = SampleCatalog()
    script = catalog.script("quote_request")
    original = script.build_scripted_models
    constructed = []

    def record_models(options):
        """Observe the actual adapter selected during catalog graph construction."""
        models = original(options)
        constructed.append(models["workflow"])
        return models

    monkeypatch.setattr(script, "build_scripted_models", record_models)
    catalog.create_run("quote_request", False, tracing=False)
    assert len(constructed) == 1
    assert isinstance(constructed[0], SimulatedModel)
    assert constructed[0].cache_reuse is False
