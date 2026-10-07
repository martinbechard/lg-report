"""Verify independent QA, missing evidence, and reproducible report exports.

Fake model receipts exercise the same callback path as providers without paid
calls. Tests protect privacy and execution totals when optional judging fails.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from qa_helpers import assessment_text, rubric_definition

from agent_runtime.harness import qa_judge
from agent_runtime.harness.argument_parser import argument_parser
from agent_runtime.harness.qa_judge import QAJudge, evidence_for
from reporting.pricing import Prices, Rate, summarize
from reporting.schema import QAEvaluation, QAVerdict, Run, Step, Usage


def verdict_text():
    """Use deliberately unequal scores to detect weighting mistakes."""
    return json.dumps({**{key: {"score": score, "reason": "Observed in step answer."}
                         for key, score in zip(("goal_achievement", "answer_quality", "speed", "cost"), (100, 80, 60, 40))},
                       "summary": "Good answer with expensive overhead."})


def fixture_run():
    """Supply captured content and a one-second metered workflow call."""
    return Run(id="sample", title="Explain caching", status="ok", steps=[
        Step(id="answer", name="assistant", kind="model", status="ok",
             start_ns=0, end_ns=1_000_000_000, provider="fixture", model="assistant",
             usage=Usage(input_tokens=10, output_tokens=2), context={"report_turn": 1},
             request=[{"role": "human", "content": "Explain caching"}],
             response=[{"role": "ai", "content": "Caching reuses saved results."}])])


def fixture_prices():
    """Keep execution and judge rates explicit and offline."""
    return Prices(as_of="2026-10-06", note="Fixture", models={
        f"fixture:{model}": Rate(input="1", output="2") for model in ("assistant", "judge")})


def install_judge(monkeypatch, text=None):
    """Capture construction policy while emitting a genuine LangChain receipt."""
    model = FakeMessagesListChatModel(responses=[AIMessage(
        content=text if text is not None else assessment_text(),
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})],
        metadata={"ls_provider": "fixture", "ls_model_name": "judge"})
    provider = Mock()
    provider.create_model.return_value = model
    monkeypatch.setattr(qa_judge, "get_provider", Mock(return_value=provider))
    monkeypatch.setattr(qa_judge, "plan_rubric",
                        lambda judge, goal, title, config=None: rubric_definition(goal))
    original = type(model)._generate
    def reply_for_evidence(messages, *args, **kwargs):
        """Keep valid fake citations aligned with actual graph-generated IDs."""
        if text is None:
            evidence = json.loads(messages[-1].content)["execution"]
            event_id = next(event["id"] for event in evidence["events"] if event["kind"] == "model")
            model.responses[0].content = assessment_text(event_id)
        return original(model, messages, *args, **kwargs)
    monkeypatch.setattr(type(model), "_generate", lambda self, *args, **kwargs: (
        reply_for_evidence(*args, **kwargs) if self is model else original(self, *args, **kwargs)))
    return provider


def test_scores_receipts_and_exports(monkeypatch, tmp_path):
    """QA is portable in JSON/HTML/Excel without inflating workflow accounting."""
    from openpyxl import load_workbook

    from reporting.excel_data import workbook_data
    from reporting.export_excel import export_workbook
    from reporting.render import render

    provider = install_judge(monkeypatch)
    run, prices = fixture_run(), fixture_prices()
    before = summarize(run, prices)
    run.qa = QAJudge({"LG_QA_PROVIDER": "fixture", "LG_QA_MODEL": "judge",
                      "LG_MODEL": "other", "LG_MAX_TOKENS": "999"}, goal="Explain caching")(run, prices)
    assert run.qa.status == "completed"
    assert run.qa.overall_score == 78.3
    assert run.qa.speed_method == "shared-rubric-v2"
    assert run.qa.verdict.speed.score == 45.4
    assert run.qa.coverage == pytest.approx(1)
    assert run.qa.cost_usd == "0.00014"
    assert run.qa.judge_steps[0].usage.input_tokens == 100
    assert not run.qa.judge_steps[0].request
    assert summarize(run, prices) == before
    assert "LG_MAX_TOKENS" not in provider.create_model.call_args.args[1]
    saved = Run.model_validate_json(run.model_dump_json())
    assert saved.qa.overall_score == 78.3
    render(saved, prices, tmp_path / "report.html")
    assert "Overall: 78.3/100" in (tmp_path / "report.html").read_text()
    export_workbook(workbook_data(saved, prices), tmp_path / "report.xlsx")
    book = load_workbook(tmp_path / "report.xlsx", data_only=True)
    assert book["QA"]["B3"].value == 78.3


def test_unknown_measurements_are_not_scores(monkeypatch):
    """A judge cannot manufacture cost when the provider supplied no usage."""
    install_judge(monkeypatch)
    run = fixture_run()
    run.steps[0].usage = None
    qa = QAJudge({})(run, fixture_prices())
    assert qa.verdict.cost.score is None
    assert qa.verdict.speed.score is None
    assert qa.coverage == pytest.approx(.7)


def test_assistant_resource_scope_excludes_input_generation(monkeypatch):
    """Slow, unpriced test-input generation cannot penalize assistant resources."""
    install_judge(monkeypatch)
    run, prices = fixture_run(), fixture_prices()
    run.steps.append(run.steps[0].model_copy(update={
        "id": "user", "model": "unpriced-user", "end_ns": 90_000_000_000,
        "context": {"model_role": "user"}, "usage": None,
    }))
    evidence = evidence_for(run, prices, "Explain caching")
    assert evidence["duration_ms"] == 1000
    assert evidence["cost_usd"] == "0.000014"
    assert evidence["model_calls"] == 1 and evidence["missing_usage"] == 0
    assert len(evidence["events"]) == 2  # Conversation context remains evidence.
    qa = QAJudge({})(run, prices)
    assert qa.measurement_scope == "assistant"
    assert qa.execution_cost_usd == "0.000014"
    assert qa.verdict.cost.score == 50


def test_static_execution_never_calls_judge(monkeypatch):
    """Explicit QA on static execution must not construct or invoke a model."""
    provider = install_judge(monkeypatch)
    run = fixture_run()
    run.demo = True
    qa = QAJudge({})(run, fixture_prices())
    assert qa.status == "skipped" and qa.overall_score is None
    assert "static/simulated" in qa.error
    assert qa.judge_steps == [] and qa.duration_ms == 0
    provider.create_model.assert_not_called()


@pytest.mark.parametrize("text", ['{}', 'not JSON', verdict_text().replace('100', '101'),
                                  verdict_text().replace('100', 'true')])
def test_bad_judgment_keeps_usage_and_no_score(monkeypatch, text):
    """A malformed reply is a QA failure, not a zero or workflow failure."""
    install_judge(monkeypatch, text)
    run = fixture_run()
    qa = QAJudge({})(run, fixture_prices())
    assert qa.status == "error" and qa.overall_score is None
    assert qa.judge_steps[0].usage.input_tokens == 100
    assert run.status == "ok"


def test_metadata_only_never_constructs_judge(monkeypatch):
    """Even explicit QA cannot bypass content capture privacy."""
    provider = install_judge(monkeypatch)
    qa = QAJudge({}, goal="Private goal", capture_content=False)(fixture_run(), fixture_prices())
    assert qa.status == "skipped" and qa.overall_score is None
    provider.create_model.assert_not_called()
    assert "Private goal" not in qa.model_dump_json()


def test_provider_failure_is_separate(monkeypatch):
    """Configuration failures preserve the workflow result and safe diagnostics."""
    provider = install_judge(monkeypatch)
    provider.create_model.side_effect = ValueError("secret credential")
    run = fixture_run()
    qa = QAJudge({})(run, fixture_prices())
    assert qa.status == "error" and qa.cost_usd is None
    assert "secret" not in qa.model_dump_json()
    assert run.status == "ok"


def test_complete_evidence_preserves_long_histories_and_all_events():
    """Normal histories must not hit per-field clipping or an event-count cap."""
    run = fixture_run()
    run.steps = [run.steps[0].model_copy(update={"id": str(i)}) for i in range(100)]
    run.steps[0].request = [{"role": "system", "content": "x" * 6000},
                            {"role": "human", "content": "Important final request"}]
    run.output = "y" * 3000
    evidence = evidence_for(run, fixture_prices(), "Explain caching")
    assert not evidence["truncated"] and len(evidence["events"]) == 100
    assert evidence["events"][0]["request"] == run.steps[0].request
    assert evidence["events"][50]["id"] == "50"
    assert evidence["output"] == run.output


def test_oversized_evidence_is_unscored_without_call(monkeypatch):
    """A total input limit cannot silently drop evidence to obtain a score."""
    provider = install_judge(monkeypatch)
    run = fixture_run()
    run.output = "x" * qa_judge.MAX_EVIDENCE_CHARS
    result = QAJudge({})(run, fixture_prices())
    assert result.status == "skipped" and result.overall_score is None
    assert not result.truncated and not result.judge_steps
    assert "exceeding the 200,000-character input limit" in result.error
    provider.create_model.assert_not_called()


def test_cli_defaults_and_sol_identity(monkeypatch):
    """QA stays off by default and does not inherit the assistant identity."""
    parser = argument_parser()
    assert parser.parse_args([]).qa is None
    assert parser.parse_args(["--qa"]).qa is True
    assert parser.parse_args(["--no-qa"]).qa is False
    provider = install_judge(monkeypatch)
    QAJudge({"LG_MODEL": "assistant", "LG_PROVIDER": "anthropic"})(fixture_run(), fixture_prices())
    qa_judge.get_provider.assert_called_once_with("codex")
    assert provider.create_model.call_args.args[0] == "gpt-6-sol"


def test_missing_dimensions_and_nonfinite_scores_rejected():
    """Require an explicit unknown per criterion rather than silent omissions."""
    data = json.loads(verdict_text())
    del data["speed"]
    with pytest.raises(ValidationError):
        QAVerdict.model_validate(data)
    data = json.loads(verdict_text())
    data["cost"]["score"] = float("nan")
    with pytest.raises(ValidationError):
        QAVerdict.model_validate(data)


def test_derived_overall_recomputed_on_load():
    """Exports cannot trust edited arithmetic that disagrees with the rubric."""
    qa = QAEvaluation(status="completed", provider="fixture", model="judge",
                      verdict=QAVerdict.model_validate_json(verdict_text()), overall_score=0)
    assert qa.overall_score == 79


@pytest.mark.parametrize("enabled", [False, True])
def test_execution_saves_optional_qa(monkeypatch, tmp_path, enabled):
    """The real recorder invokes QA once after execution, or never when disabled."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    from reporting.execute_runnable import execute_runnable

    provider = install_judge(monkeypatch)
    assistant = FakeMessagesListChatModel(responses=[AIMessage(content="An answer")])

    def answer(state, config):
        """Generate callbacks through the same LangGraph path as the samples."""
        return {"messages": [assistant.invoke(state["messages"], config=config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    execute_runnable(graph.compile(), {"messages": [("human", "Explain caching")]},
                     tmp_path / "run", fixture_prices(), provider="fixture", model="assistant",
                     include_output=True, qa=QAJudge({}) if enabled else None)
    saved = Run.model_validate_json((tmp_path / "run/run.json").read_text())
    assert saved.status == "ok"
    assert (saved.qa is not None) is enabled
    assert provider.create_model.call_count == int(enabled)
    assert len([step for step in saved.steps if step.kind == "model"]) == 1


@pytest.mark.parametrize("flag,expected", [([], True), (["--no-qa"], False)])
def test_settings_resolve_env_and_cli(monkeypatch, tmp_path, flag, expected):
    """Explicit disable wins over the environment without constructing a model."""
    from agent_runtime.harness.settings import settings_for

    env = tmp_path / "sample.env"
    env.write_text("LG_QA=true\nLG_QA_MODEL=file-model\n")
    monkeypatch.delenv("LG_QA_MODEL", raising=False)
    monkeypatch.delenv("LG_QA", raising=False)
    from agent_runtime.harness import settings

    monkeypatch.setattr(settings, "get_prices", lambda *args, **kwargs: fixture_prices())
    args = argument_parser().parse_args(["--static", "--env-file", str(env),
                                        "--qa-model", "chosen-model", *flag])
    result = settings_for(__file__, "QA test", args=args)
    assert (result.qa is not None) is expected
    if expected:
        assert result.qa.values["LG_QA_MODEL"] == "chosen-model"


def test_web_session_judges_selected_goal(monkeypatch, tmp_path):
    """Browser selection and turn saving share the same independent judge."""
    from fastapi.testclient import TestClient

    from agent_runtime.harness.sample_catalog import SampleCatalog
    from agent_runtime.web.server import create_app

    provider = install_judge(monkeypatch)
    catalog = SampleCatalog()
    def live_fixture(selected, live):
        """Use a local graph fixture while testing a live session's QA policy."""
        return catalog.create_run(selected, False)

    app = create_app(directory=tmp_path, catalog=catalog, prices=fixture_prices(),
                     default_live=True, factory=live_fixture,
                     capture_content=True, qa=QAJudge({}, goal="Wrong launch goal"))
    with TestClient(app, base_url="http://localhost") as client:
        thread = client.post("/api/sessions", json={"sample": "simple_chat"}).json()["threadId"]
        session = app.state.sessions[thread]
        assert session.qa.goal == catalog.get("simple_chat").goal
        response = client.post(f"/api/chat/{thread}", json={
            "threadId": thread, "runId": "11111111-1111-4111-8111-111111111111", "state": {},
            "messages": [{"id": "prompt", "role": "user", "content": "Explain the main steps in an agent workflow."}],
            "tools": [], "context": [], "forwardedProps": {"prompt": "Explain the main steps in an agent workflow."}})
        assert response.status_code == 200 and 'RUN_FINISHED' in response.text
        saved = Run.model_validate_json((tmp_path / "simple_chat/run.json").read_text())
        assert saved.qa.status == "completed"
        assert saved.qa.goal == catalog.get("simple_chat").goal
        assert provider.create_model.call_count == 1


def test_failed_execution_survives_failed_qa(monkeypatch, tmp_path):
    """Optional QA cannot replace the workflow's exception or failure status."""
    from langchain_core.runnables import RunnableLambda

    from reporting.execute_runnable import execute_runnable

    provider = install_judge(monkeypatch, "invalid judgment")
    assistant = FakeMessagesListChatModel(responses=[AIMessage(content="Partial answer")])

    def fail_after_answer(inputs, config):
        """Record partial evidence, then fail at the workflow boundary."""
        assistant.invoke("Question", config=config)
        raise RuntimeError("workflow failure")

    with pytest.raises(RuntimeError, match="workflow failure"):
        execute_runnable(RunnableLambda(fail_after_answer), {}, tmp_path / "run",
                         fixture_prices(), provider="fixture", model="assistant",
                         include_output=True, qa=QAJudge({}))
    saved = Run.model_validate_json((tmp_path / "run/run.json").read_text())
    assert saved.status == "error" and saved.qa.status == "error"
    assert provider.create_model.call_count == 1
