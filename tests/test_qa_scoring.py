"""Protect shared QA criteria from discretionary scores and missing evidence.

These checks exercise observable classifications, numerical monotonicity, and
rubric generation without real model calls or changing execution measurements.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from qa_helpers import assessment_text, rubric_definition

from agent_runtime.harness.qa_judge import QAJudge, plan_rubric
from reporting.qa_scoring import score_assessment
from reporting.schema import QAAssessment, QARubric, Run, Step, Usage


def execution():
    """Provide an Agent call with independent latency and output quantities."""
    return Run(id="run", title="Example", status="ok", steps=[
        Step(id="answer", name="Agent", kind="model", status="ok", start_ns=0,
             end_ns=5_000_000_000, context={"report_turn": 1},
             usage=Usage(input_tokens=10, output_tokens=125))])


def test_same_completed_goal_always_gets_full_credit():
    """A different example or explanation cannot turn all-met checks into 96."""
    rubric = rubric_definition()
    assessment = QAAssessment.model_validate_json(assessment_text())
    first = score_assessment(rubric, assessment, execution(), ".000014")
    assessment.goal_achievement[0].reason = "A hypothetical flight observation illustrates the requested concept."
    second = score_assessment(rubric, assessment, execution(), ".000014")
    assert first.goal_achievement.score == second.goal_achievement.score == 100
    assert first.answer_quality.score == 80
    assert first.speed.score == first.cost.score == 50


@pytest.mark.parametrize("outcome,score", [("unmet", 0), ("unknown", None)])
def test_missing_goal_evidence_cannot_earn_full_credit(outcome, score):
    """An observed omission and absent evidence retain distinct meanings."""
    assessment = QAAssessment.model_validate_json(assessment_text())
    assessment.goal_achievement[0].outcome = outcome
    assert score_assessment(rubric_definition(), assessment, execution(), None).goal_achievement.score == score


@pytest.mark.parametrize("defect", ["partial_goal", "duplicate", "missing", "foreign_id", "empty_citation"])
def test_invalid_classifications_fail_closed(defect):
    """The judge cannot add, omit, duplicate, or grant unsupported partial points."""
    assessment = QAAssessment.model_validate_json(assessment_text())
    if defect == "partial_goal":
        assessment.goal_achievement[0].outcome = "partial"
    elif defect == "duplicate":
        assessment.answer_quality[1] = assessment.answer_quality[0]
    elif defect == "missing":
        assessment.answer_quality.pop()
    elif defect == "foreign_id":
        assessment.goal_achievement[0].evidence_ids = ["invented"]
    else:
        assessment.goal_achievement[0].evidence_ids = []
    with pytest.raises(ValueError):
        score_assessment(rubric_definition(), assessment, execution(), ".01")


@pytest.mark.parametrize("defect", ["sum", "duplicate", "partial", "zero_anchor"])
def test_invalid_rubric_rejected(defect):
    """Reject broken scoring contracts before any execution is assessed."""
    value = rubric_definition().model_dump()
    if defect == "sum":
        value["goal_achievement"][0]["points"] = 96
    elif defect == "duplicate":
        value["answer_quality"][0]["id"] = "goal"
    elif defect == "partial":
        value["goal_achievement"][0]["partial_when"] = "Almost done"
    else:
        value["cost"]["usd_at_half_score"] = 0
    with pytest.raises(ValidationError):
        QARubric.model_validate(value)


def test_speed_and_cost_are_monotonic_and_preserve_unknowns():
    """Faster or cheaper work cannot score lower under the frozen anchors."""
    rubric, assessment, run = rubric_definition(), QAAssessment.model_validate_json(assessment_text()), execution()
    first = score_assessment(rubric, assessment, run, ".000014")
    run.steps[0].end_ns *= 2
    slower = score_assessment(rubric, assessment, run, ".000028")
    assert slower.speed.score < first.speed.score
    assert slower.cost.score < first.cost.score
    run.steps[0].usage = None
    missing = score_assessment(rubric, assessment, run, None)
    assert missing.speed.score is missing.cost.score is None


def test_rubric_creation_has_no_candidate_results():
    """Shared anchors are chosen before the judge sees any candidate evidence."""
    rubric = rubric_definition()
    judge = Mock()
    judge.invoke.return_value = AIMessage(content=rubric.model_dump_json())
    result = plan_rubric(judge, rubric.goal, "Task")
    prompt = judge.invoke.call_args.args[0]
    assert json.loads(prompt[1].content) == {"title": "Task", "goal": rubric.goal}
    assert "hypothetical example" in prompt[0].content
    assert result.fingerprint == rubric.fingerprint
    judge.invoke.return_value = AIMessage(content=rubric.model_copy(update={"goal": "Changed"}).model_dump_json())
    with pytest.raises(ValueError, match="exact supplied goal"):
        plan_rubric(judge, rubric.goal, "Task")


def test_supplied_rubric_reused_without_regeneration(monkeypatch, tmp_path):
    """Every evaluation loads the identical contract rather than replanning it."""
    from test_qa_judge import fixture_prices, fixture_run, install_judge

    install_judge(monkeypatch)
    planner = Mock(side_effect=AssertionError("Must not regenerate shared criteria"))
    monkeypatch.setattr("agent_runtime.harness.qa_judge.plan_rubric", planner)
    rubric = rubric_definition()
    path = tmp_path / "rubric.json"
    path.write_text(rubric.model_dump_json())
    judge = QAJudge({"LG_QA_RUBRIC": str(path)}, goal=rubric.goal)
    results = [judge(fixture_run(), fixture_prices()) for _ in range(2)]
    assert all(result.status == "completed" for result in results)
    assert {result.rubric.fingerprint for result in results} == {rubric.fingerprint}
    planner.assert_not_called()


def test_standalone_turns_share_first_plan(monkeypatch):
    """Repeated saved turns cannot acquire fresh criteria for the same goal."""
    from test_qa_judge import fixture_prices, fixture_run, install_judge

    install_judge(monkeypatch)
    planner = Mock(return_value=rubric_definition())
    monkeypatch.setattr("agent_runtime.harness.qa_judge.plan_rubric", planner)
    judge = QAJudge({}, goal="Explain caching")
    assert judge(fixture_run(), fixture_prices()).status == "completed"
    assert judge(fixture_run(), fixture_prices()).status == "completed"
    planner.assert_called_once()


def test_reports_show_one_shared_contract_and_detect_mismatch(tmp_path):
    """A shared judge label must not conceal different rubric content."""
    from test_qa_judge import fixture_prices, fixture_run

    from reporting.compare import render_comparison
    from reporting.render import render
    from reporting.schema import QAEvaluation

    paths = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        run, prices = fixture_run(), fixture_prices()
        rubric = rubric_definition()
        rubric.goal_achievement[0].description = "Explain <script>untrusted</script>"
        assessment = QAAssessment.model_validate_json(assessment_text())
        run.qa = QAEvaluation(status="completed", provider="fixture", model="judge", rubric=rubric,
                              assessment=assessment, verdict=score_assessment(rubric, assessment, run, ".01"))
        path = directory / "run.json"
        path.write_text(run.model_dump_json())
        (directory / "prices.json").write_text(prices.model_dump_json())
        render(run, prices, directory / "report.html")
        paths.append(path)
    destination = tmp_path / "comparison.html"
    render_comparison(paths, destination)
    html = destination.read_text()
    assert html.count("<summary>Scoring criteria</summary>") == 1
    assert "&lt;script&gt;untrusted&lt;/script&gt;" in html
    assert "<script>untrusted</script>" not in html
    run.qa.rubric.cost.usd_at_half_score = .1
    paths[-1].write_text(run.model_dump_json())
    render_comparison(paths, destination)
    assert "different scoring criteria" in destination.read_text()


@pytest.mark.parametrize("source", ["sample", "environment", "object"])
def test_sample_rubric_precedence(monkeypatch, tmp_path, source):
    """Explicit inputs override a sample contract without invoking the planner."""
    from test_qa_judge import fixture_prices, fixture_run, install_judge

    install_judge(monkeypatch)
    planner = Mock(side_effect=AssertionError("Saved criteria must not be regenerated"))
    monkeypatch.setattr("agent_runtime.harness.qa_judge.plan_rubric", planner)
    selected = rubric_definition()
    sample_path = tmp_path / "qa-rubric.json"
    sample_path.write_text(selected.model_dump_json())
    values = {}
    supplied = None
    if source != "sample":
        # A broken lower-priority file must not interfere with an override.
        sample_path.write_text("invalid JSON")
        if source == "environment":
            explicit = tmp_path / "explicit.json"
            explicit.write_text(selected.model_dump_json())
            values["LG_QA_RUBRIC"] = str(explicit)
        else:
            supplied = selected
            values["LG_QA_RUBRIC"] = str(tmp_path / "missing.json")
    result = QAJudge(values, goal=selected.goal, rubric=supplied,
                     sample_rubric_path=sample_path)(fixture_run(), fixture_prices())
    assert result.status == "completed"
    assert result.rubric.fingerprint == selected.fingerprint
    planner.assert_not_called()


@pytest.mark.parametrize("contents", ["not JSON", rubric_definition("Old goal").model_dump_json()])
def test_invalid_sample_rubric_does_not_generate_replacement(monkeypatch, tmp_path, contents):
    """Malformed criteria or goal drift fail visibly without changing the contract."""
    from test_qa_judge import fixture_prices, fixture_run, install_judge

    install_judge(monkeypatch)
    planner = Mock(side_effect=AssertionError("Must not replace invalid saved criteria"))
    monkeypatch.setattr("agent_runtime.harness.qa_judge.plan_rubric", planner)
    path = tmp_path / "qa-rubric.json"
    path.write_text(contents)
    result = QAJudge({}, goal="Explain caching", sample_rubric_path=path)(fixture_run(), fixture_prices())
    assert result.status == "error"
    assert result.error
    planner.assert_not_called()


def test_all_sample_rubrics_match_declared_goals():
    """Every shipped lesson has a valid frozen contract for its current goal."""
    from agent_runtime.harness.qa_judge import load_rubric
    from agent_runtime.harness.sample_catalog import SampleCatalog

    for sample in SampleCatalog().samples.values():
        assert sample.qa_rubric_path is not None, sample.id
        rubric = load_rubric(sample.qa_rubric_path, sample.goal or sample.description)
        assert rubric.goal == (sample.goal or sample.description)


def test_sample_without_rubric_allows_generation(tmp_path):
    """Discovery represents an absent optional file without inventing a contract."""
    from agent_runtime.harness.sample_catalog import Sample

    sample = Sample(id="new", name="New", description="Explain caching",
                    implementation="new", directory=tmp_path)
    assert sample.qa_rubric_path is None


def test_launch_binds_selected_samples_rubric(monkeypatch, tmp_path):
    """Console and batch setup select the lesson's rubric without loading models."""
    from test_qa_judge import fixture_prices

    from agent_runtime.harness import settings
    from agent_runtime.harness.argument_parser import argument_parser
    from agent_runtime.harness.sample_catalog import SampleCatalog

    catalog = SampleCatalog()
    launch = settings.Settings(live=True, output=tmp_path, prices=fixture_prices(),
                               capture_content=True, overwrite=True, qa=QAJudge({}))
    monkeypatch.setattr(settings, "settings_for", lambda *args, **kwargs: launch)
    args = argument_parser().parse_args(["--live", "--qa"])
    # The application normally resolves repeated --option arguments before setup.
    args.options = {}
    for name in ("simple_chat", "tool_chat"):
        selected, _, _ = settings.prepare_sample(catalog, name, args)
        assert selected.qa.goal == catalog.get(name).goal
        assert selected.qa.sample_rubric_path == catalog.get(name).qa_rubric_path
    assert launch.qa.sample_rubric_path is None
