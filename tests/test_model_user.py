"""Verify live user-role selection, adaptive dialogue, and recorded accounting.

Fake models expose the exact scenario/history sent to the user role without
provider calls. A real conversation/recorder test proves usage attribution and
comparison exclusion rather than relying only on configuration assertions.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import os
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_runtime.harness.argument_parser import parse_arguments
from agent_runtime.harness.console_client import ConsoleClient
from agent_runtime.harness.model_user import (
    ModelUserClient,
    build_user_model,
    user_scenario,
)
from agent_runtime.harness.sample_catalog import SampleCatalog


@pytest.fixture(autouse=True)
def isolated_environment():
    """Never pick up the workstation's model selection or provider credentials."""
    with patch.dict(os.environ, {}, clear=True):
        yield


def decision(done, reason="The requested objectives are covered.", message=""):
    """Author explicit model assessments rather than implicit magic stop tokens."""
    return json.dumps({"done": done, "reason": reason, "message": message})


def user_client(replies, **kwargs):
    """Construct one independent conversation with observable terminal output."""
    sink = ConsoleClient(write=Mock(), answer=Mock(return_value="reject"))
    model = FakeMessagesListChatModel(responses=[AIMessage(content=text) for text in replies])
    return ModelUserClient(sink, model, initial_request='{"goal":"test"}',
                           scenario="A user wants to understand caching.", **kwargs)


def test_adaptive_followup_uses_real_reply_and_stops_at_limit():
    """Follow-ups consume actual answers; the initial structured request survives."""
    client = user_client([decision(False, "Expiry remains unclear.", "What if the cache expires?"), decision(False, "Expiry was not answered.")], max_turns=2)
    assert client.receive().prompt == '{"goal":"test"}'
    client.respond({"messages": [AIMessage(content="The cache lasts five minutes.")]})
    with patch.object(type(client.model), "invoke", wraps=client.model.invoke) as invoke:
        assert client.receive().prompt == "What if the cache expires?"
        messages = invoke.call_args.args[0]
        assert any("five minutes" in message.content for message in messages)
        assert "caching" in messages[0].content
        assert invoke.call_args.kwargs["config"]["metadata"]["model_role"] == "user"
    assert client.receive() is None


def test_end_and_provider_failure_never_replay_script():
    """Ending is explicit; errors propagate instead of manufacturing user input."""
    client = user_client([decision(True)])
    client.receive()
    assert client.receive() is None
    assert client.receive() is None
    client = user_client(["unused"])
    client.receive()
    with patch.object(type(client.model), "invoke", side_effect=RuntimeError("provider failed")), pytest.raises(RuntimeError, match="provider failed"):
        client.receive()


def test_clarification_is_adaptive_but_approval_uses_existing_policy():
    """Generated dialogue cannot grant approval in place of the explicit policy."""
    client = user_client(["Put the addresses on the envelopes."])
    client.receive()
    assert client.answer({"kind": "question", "text": "Where do the addresses go?"}) == "Put the addresses on the envelopes."
    assert any("Where do the addresses" in message.content for message in client.history)
    assert client.answer({"kind": "approval"}) == "reject"
    client.client.answer_callback.assert_called_once_with({"kind": "approval"})
    client.questions = 10
    with pytest.raises(ValueError, match="ten clarification"):
        client.answer({"kind": "question"})


def test_empty_and_tool_responses_fail_explicitly():
    """A text-only user contract never forwards model tool calls as user consent."""
    client = user_client(["   "])
    client.receive()
    with pytest.raises(ValueError, match="empty"):
        client.receive()
    tool = AIMessage(content="", tool_calls=[{"name": "write", "args": {}, "id": "call"}])
    with patch.object(type(client.model), "invoke", return_value=tool), pytest.raises(ValueError, match="tool calls"):
        client.receive()


def test_scenario_is_loaded_from_user_side_of_sample():
    """Existing scenario facts guide the persona; canned assistant replies do not."""
    catalog = SampleCatalog()
    prompt = user_scenario(catalog, "quote_request", catalog.prompts("quote_request"))
    assert "500 households" in prompt
    assert "quote_interpreter" not in prompt
    assert "QuoteDecision" not in prompt


@pytest.mark.parametrize("assistant_provider,assistant_model", [("openai", "gpt-5.5"), ("codex", "gpt-6-sol"), ("copilot", "another")])
def test_user_provider_does_not_follow_tested_model(assistant_provider, assistant_model):
    """One explicit user identity/options stays fixed while target providers vary."""
    values = {"LG_PROVIDER": assistant_provider, "LG_MODEL": assistant_model,
              "LG_MAX_TOKENS": "999", "LG_EFFORT": "high", "LG_AVAILABLE_MODELS": assistant_model,
              "LG_USER_MODEL": "gpt-6-luna"}
    model = build_user_model(values)
    assert model.model_name == "gpt-6-luna" and model.provider == "codex"
    assert model.reasoning_effort == "high"


def test_user_can_use_native_api_adapter():
    """The same provider protocol supports a different user transport explicitly."""
    from langchain_openai import ChatOpenAI

    model = build_user_model({"LG_USER_MODEL": "gpt-6-luna", "LG_USER_PROVIDER": "openai",
                              "OPENAI_API_KEY": "placeholder", "LG_USER_MAX_TOKENS": "321"})
    assert isinstance(model, ChatOpenAI) and model.model_name == "gpt-6-luna"
    assert model.max_tokens == 321


def test_cli_defaults_and_demo_selection(monkeypatch):
    """Live mode selects Luna automatically; static mode keeps fixed text."""
    catalog = SampleCatalog()
    _, args = parse_arguments(catalog, ["--live"])
    assert (args.user_model, args.user_provider, args.user_turns, args.client) == ("gpt-6-luna", "codex", 3, "agent")
    monkeypatch.setenv("LG_USER_MODEL", "gpt-6-luna")
    _, args = parse_arguments(catalog, ["--static"])
    assert not args.live and args.client == "static"
    _, args = parse_arguments(catalog, ["--live", "--sample", "quote_request", "--client", "agent"])
    assert args.user_model == "gpt-6-luna"
    with pytest.raises(SystemExit):
        parse_arguments(catalog, ["--live", "--user-model", "--user-turns", "0"])


def test_live_recording_distinguishes_user_cost_and_comparison(tmp_path):
    """User calls flow through real callbacks and remain outside target charts."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    from agent_runtime.harness.conversation import Conversation
    from reporting.compare import comparison_entry, render_comparison
    from reporting.exchange import ExchangeRate
    from reporting.execute_runnable import execute_runnable
    from reporting.pricing import Prices, Rate
    from reporting.schema import Run

    def response(text):
        """Supply metered messages to exercise receipt normalization."""
        return AIMessage(content=text, usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})

    model = FakeMessagesListChatModel(responses=[response("A real answer")],
                                    metadata={"ls_provider": "fixture", "ls_model_name": "assistant"})
    def answer(state, config):
        """Use the same graph invocation path as ordinary live sample models."""
        return {"messages": [model.invoke(state["messages"], config=config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer"); graph.add_edge("answer", END)
    client = ModelUserClient(ConsoleClient(write=lambda _: None),
        FakeMessagesListChatModel(responses=[response(decision(False, "Need more detail.", "An adaptive follow-up")), response(decision(True))],
                                  metadata={"ls_provider": "fixture", "ls_model_name": "user"}),
        initial_request="Explain caching", scenario="Ask about caching", max_turns=2)
    prices = Prices(as_of="2026-10-06", note="Fixture",
                    exchange=ExchangeRate(rate="0.8", date="2026-10-05"),
                    models={f"fixture:{role}": Rate(input="1", output="2") for role in ("assistant", "user")})
    out = tmp_path / "run"
    execute_runnable(Conversation(graph.compile(), client), {}, out, prices,
                     provider="fixture", model="assistant", include_output=True)
    run = Run.model_validate_json((out / "run.json").read_text())
    users = [step for step in run.steps if step.context.get("model_role") == "user"]
    assert len(users) == 2 and users[0].model == "user" and users[0].usage.input_tokens == 10
    assert users[0].context["report_turn"] == 1
    entry = comparison_entry(run, prices, "run.json")
    assert entry["summary"]["model_calls"] == 2 and entry["user_summary"]["model_calls"] == 2
    assert entry["models"] == ["fixture:assistant"] and entry["user_models"] == ["fixture:user"]
    assert entry["user_test"]["status"] == "goal_met"
    assert entry["user_test"]["reason"] == "The requested objectives are covered."
    assert "Why it stopped:" in (out / "report.html").read_text()
    assert len(entry["chart"]["bars"]) == 2
    assert all(call.context.get("model_role") != "user" for call in entry["calls"])
    comparison = tmp_path / "comparison.html"
    render_comparison([out / "run.json", out / "run.json"], comparison)
    assert "Goal &amp; why it stopped" in comparison.read_text()
    assert "The requested objectives are covered." in comparison.read_text()


@pytest.mark.parametrize("live,mode,expected", [(True, "agent", True), (True, "static", False), (False, "static", False), (True, "console", False)])
def test_launch_only_wraps_configured_live_agent_client(monkeypatch, live, mode, expected):
    """Configuration alone cannot replace a human or invoke a model in demo mode."""
    from types import SimpleNamespace

    from agent_runtime.harness import execute_conversation, model_user, settings
    from agent_runtime.harness.console_application import ConsoleApplication

    catalog = SampleCatalog()
    _, args = parse_arguments(catalog, ["--live" if live else "--static", "--client", mode, "--user-model"])
    monkeypatch.setattr(settings, "prepare_sample", lambda *args: (SimpleNamespace(live=live), {}, None))
    model = Mock()
    monkeypatch.setattr(model_user, "build_user_model", model)
    execute = Mock()
    monkeypatch.setattr(execute_conversation, "execute_conversation", execute)
    ConsoleApplication(catalog, args).run_session("simple_chat", ConsoleClient())
    assert isinstance(execute.call_args.kwargs["client"], ModelUserClient) is expected
    assert model.call_count == int(expected)


def test_batch_quote_uses_adaptive_client_without_terminal(tmp_path, monkeypatch):
    """The batch launcher honors the same opt-in instead of hanging on stdin."""
    import runpy
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run_sample = runpy.run_path(str(root / "scripts/run_samples.py"))["run_sample"]
    calls = []

    def child(command, **kwargs):
        """Capture the launch contract and provide the expected export outputs."""
        calls.append((command, kwargs))
        for name in ("report.html", "report.xlsx"):
            (tmp_path / name).write_text("fixture")

    monkeypatch.setattr(subprocess, "run", child)
    run_sample("quote_request", tmp_path, prices=root / "models.json", fx_file=None,
               env={"LG_USER_MODEL": "gpt-6-luna"})
    command, options = calls[0]
    assert command[command.index("--client") + 1] == "agent"
    assert options["stdin"] == subprocess.DEVNULL


def test_structured_sample_keeps_one_top_level_request(monkeypatch):
    """Clarifications adapt within the structured seed, not a second JSON request."""
    from types import SimpleNamespace

    from agent_runtime.harness import execute_conversation, model_user, settings
    from agent_runtime.harness.console_application import ConsoleApplication

    catalog = SampleCatalog()
    _, args = parse_arguments(catalog, ["--live", "--sample", "quote_request", "--user-model"])
    monkeypatch.setattr(settings, "prepare_sample", lambda *args: (SimpleNamespace(live=True), {}, None))
    monkeypatch.setattr(model_user, "build_user_model", lambda _: FakeMessagesListChatModel(responses=[AIMessage(content=decision(True))]))
    execute = Mock()
    monkeypatch.setattr(execute_conversation, "execute_conversation", execute)
    ConsoleApplication(catalog, args).run_session("quote_request", ConsoleClient())
    client = execute.call_args.kwargs["client"]
    assert client.receive().prompt == catalog.prompts("quote_request")[0]
    assert client.receive() is None


def test_goal_allows_one_turn_despite_long_sample():
    """Explicit completion criteria replace the authored conversation length."""
    client = user_client([decision(True, "Both workflow and observation were explained.")],
                         goal="Explain workflow and observations", minimum_turns=5)
    client.receive()
    assert client.receive() is None
    assert client.stop_outcome["user_test_status"] == "goal_met"
    assert client.stop_outcome["user_test_turns"] == 1
    assert "no minimum turn count" in client.system.content


def test_without_goal_requires_sample_length_and_objectives():
    """A generic satisfaction claim cannot bypass the goal-free turn minimum."""
    client = user_client([decision(True)], minimum_turns=2)
    client.receive()
    with pytest.raises(ValueError, match="minimum turns"):
        client.receive()
    assert client.stop_outcome is None


def test_cap_assesses_unmet_goal_and_cannot_issue_followup():
    """The hard safety cap stays distinct from the user model's success judgment."""
    client = user_client([decision(False, "The tool observation objective is missing.", "Explain observations.")],
                         goal="Explain workflow and observations", max_turns=1)
    client.receive()
    assert client.receive() is None
    assert client.stop_outcome["user_test_status"] == "turn_limit"
    assert "missing" in client.stop_outcome["user_test_reason"]
    assert client.receive() is None


@pytest.mark.parametrize("raw", ["[END]", '{"done":true,"message":""}',
                                '{"done":"true","reason":"yes","message":""}',
                                decision(False, "Still missing an objective.")])
def test_invalid_decision_cannot_silently_end(raw):
    """Require explicit valid evidence and a usable follow-up when continuing."""
    client = user_client([raw])
    client.receive()
    with pytest.raises(ValueError):
        client.receive()
    assert not client.finished


def test_cancel_records_explanation_without_new_turn():
    """Cancellation never becomes goal success even if the explanation is positive."""
    client = user_client(["The user cancelled before providing required quote details."])
    client.receive()
    client.respond({"status": "cancelled", "messages": [AIMessage(content="Cancelled")]})
    assert client.receive() is None
    assert client.stop_outcome["user_test_status"] == "cancelled"
    assert "required quote details" in client.stop_outcome["user_test_reason"]


@pytest.mark.parametrize("capture_content", [False, True])
def test_outcome_trace_respects_content_policy(tmp_path, capture_content):
    """Private scenario text stays out of metadata-only evidence and HTML."""
    from agent_runtime.harness.trace_capture import TraceCapture
    from reporting.normalize import normalize
    from reporting.render import user_test_outcome

    path = tmp_path / "spans.jsonl"
    capture = TraceCapture(path, "fixture", "user", capture_content=capture_content)
    client = user_client([decision(True, "Secret objective satisfied.")], goal="Secret objective")
    client.set_run_config({"callbacks": [capture]})
    client.receive()
    client.receive()
    capture.close()
    run = normalize(path, title="Privacy test")
    outcome = user_test_outcome(run)
    assert outcome["status"] == "goal_met" and outcome["turns"] == 1
    assert ("Secret" in path.read_text()) is capture_content
    assert (outcome["reason"] == "Secret objective satisfied.") is capture_content


def test_all_shipped_samples_define_stop_goals():
    """Every selectable variant owns explicit criteria even when code is shared."""
    catalog = SampleCatalog()
    assert all(sample.goal and sample.goal.strip() for sample in catalog.samples.values())
    assert catalog.get("nested_workflows").goal != catalog.get("nested_workflows_review_limit").goal


@pytest.mark.parametrize("mode,provider,expected_live,expected_client", [
    ([], "openai", False, "static"),
    (["--demo"], "codex", True, "agent"),
    (["--demo"], "copilot", True, "agent"),
    (["--demo", "--static"], "codex", False, "static"),
    (["--demo", "--live"], "openai", True, "agent"),
    (["--live", "--client", "static"], "openai", True, "static"),
    (["--live", "--client", "console"], "openai", True, "console"),
])
def test_demo_is_unattended_independent_of_model_mode(tmp_path, monkeypatch, mode, provider, expected_live, expected_client):
    """Demo chooses who supplies input, while static/live choose model behavior."""
    env = tmp_path / "empty.env"
    env.write_text("")
    monkeypatch.setenv("LG_PROVIDER", provider)
    # Local login adapters require an explicit assistant model independently of
    # the default user model; selection must not manufacture an assistant ID.
    monkeypatch.setenv("LG_MODEL", "gpt-6-sol")
    _, args = parse_arguments(SampleCatalog(), ["--env-file", str(env), *mode])
    assert (args.live, args.client) == (expected_live, expected_client)
    if expected_client == "agent":
        assert args.user_model == "gpt-6-luna" and args.user_provider == "codex"


@pytest.mark.parametrize("flags", [
    ["--demo", "--live", "--client", "console"],
    ["--demo", "--static", "--client", "angular"],
    ["--static", "--client", "agent"],
    ["--live", "--static"],
])
def test_incompatible_mode_choices_fail_before_execution(flags):
    """Do not promise an unattended run that could block on a person or fake LLM."""
    with pytest.raises(SystemExit):
        parse_arguments(SampleCatalog(), flags)


@pytest.mark.parametrize("provider", ["codex", "copilot", "openai", "anthropic"])
def test_live_user_provider_override_uses_shared_adapter(tmp_path, monkeypatch, provider):
    """CLI and environment select the same registry without new SDK-specific paths."""
    from agent_runtime.harness import model_user

    env = tmp_path / "user.env"
    env.write_text("LG_USER_PROVIDER=copilot\nLG_USER_MODEL=config-model\n")
    _, args = parse_arguments(SampleCatalog(), ["--demo", "--live", "--env-file", str(env),
                                               "--user-provider", provider, "--user-model", "chosen-model"])
    adapter = Mock()
    registry = Mock(return_value=adapter)
    monkeypatch.setattr(model_user, "get_provider", registry)
    result = build_user_model({"LG_USER_PROVIDER": args.user_provider, "LG_USER_MODEL": args.user_model})
    registry.assert_called_once_with(provider)
    assert result is adapter.create_model.return_value
    assert adapter.create_model.call_args.args[0] == "chosen-model"
    assert adapter.create_model.call_args.args[1]["LG_EFFORT"] == "high"
