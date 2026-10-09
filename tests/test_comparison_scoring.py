"""Verify bell-curve QA arithmetic against known percentiles and saved evidence.

Population membership, missing measurements, ties, and reordering must never
invent costs, lose judge assessments, or change original recordings. These
small offline fixtures exercise actual reporting projections, without models.
AI assistance: Codex. Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import pytest

from reporting.compare import comparison_entry
from reporting.comparison_scoring import distribution, percentile, score_comparison
from reporting.pricing import Prices, Rate
from reporting.schema import QAEvaluation, QAScore, QAVerdict, Run, Step, Usage


def entry(name, cost=1, rate=20, seconds=10):
    """Set independent resource measurements via real receipts and fixed tariffs."""
    run = Run(id=name, title=name, status='ok', steps=[Step(
        id='call', name='Agent', kind='model', status='ok', start_ns=0,
        end_ns=seconds * 1_000_000_000, provider='fixture', model=name,
        usage=Usage(input_tokens=cost * 1_000_000, output_tokens=rate * seconds),
        context={'report_turn': 1},
    )], qa=QAEvaluation(status='completed', provider='fixture', model='judge',
                       verdict=QAVerdict(goal_achievement=QAScore(score=80, reason='goal'),
                                         answer_quality=QAScore(score=60, reason='quality'),
                                         speed=QAScore(score=99, reason='old speed'),
                                         cost=QAScore(score=99, reason='old cost'), summary='assessment')))
    prices = Prices(as_of='2026-10-07', note='Fixture',
                    models={f'fixture:{name}': Rate(input='1', output='0')})
    return comparison_entry(run, prices, name)


def test_population_percentiles_direction_ties_and_missing():
    """Known standard-normal positions define the scale, including zero variance."""
    population = distribution([1, 2, 3, None])
    assert population.count == 3 and population.mean == 2
    assert population.stddev == pytest.approx((2 / 3) ** .5)
    assert percentile(2, population) == 50
    assert percentile(1, population) == pytest.approx(11.0335681)
    assert percentile(1, population, lower_is_better=True) == pytest.approx(88.9664319)
    assert percentile(1, distribution([1, 1])) == 50
    assert percentile(1, distribution([1])) is None
    assert percentile(None, population) is None
    assert percentile(0, distribution([])) is None


def test_cohort_scores_reweight_and_preserve_source_judgments():
    """Shorter elapsed time scores higher; revised weights determine totals."""
    entries = [entry('a', 1, 10, 30), entry('b', 2, 20, 20), entry('c', 3, 30, 10)]
    originals = [e['run'] for e in entries]
    before = [r.model_dump_json() for r in originals]
    score_comparison(entries)
    assert [e['run'].qa.verdict.cost.score for e in entries] == [89.0, 50.0, 11.0]
    assert [e['run'].qa.verdict.speed.score for e in entries] == [11.0, 50.0, 89.0]
    assert [e['run'].qa.overall_score for e in entries] == [73.7, 62.0, 50.3]
    assert [r.model_dump_json() for r in originals] == before
    for e in entries:
        qa = e['run'].qa
        assert qa.verdict.goal_achievement.score == 80 and qa.verdict.answer_quality.score == 60
        assert qa.verdict.summary == 'assessment'
        assert qa.comparison_scoring.cost.count == 3
        assert Run.model_validate_json(e['run'].model_dump_json()).qa == qa
    score_comparison(entries)
    assert [e['run'].qa.verdict.cost.score for e in entries] == [89.0, 50.0, 11.0]
    score_comparison(entries[::-1])
    assert [e['run'].qa.verdict.speed.score for e in entries] == [11.0, 50.0, 89.0]


def test_unknown_static_and_unjudged_population_membership():
    """Only measured live data enters distributions; absence is not a cheap run."""
    entries = [entry('a'), entry('b'), entry('missing'), entry('static'), entry('unjudged')]
    entries[2]['summary']['unpriced_calls'] = 1
    entries[2]['performance'] = {'tokens_per_second': None, 'model_seconds_per_turn': None, 'agent_elapsed_seconds': None}
    entries[3]['run'].demo = True
    entries[3]['performance'] = {'tokens_per_second': None, 'model_seconds_per_turn': None, 'agent_elapsed_seconds': None}
    entries[4]['run'].qa = None
    basis = score_comparison(entries)
    assert basis.cost.count == basis.elapsed_seconds.count == 3
    assert entries[2]['run'].qa.verdict.cost.score is None
    assert entries[2]['run'].qa.verdict.speed.score is None
    assert entries[2]['run'].qa.coverage == .5
    assert entries[3]['run'].qa.speed_method == 'judge-v1'
    assert entries[4]['run'].qa is None
    assert entries[0]['run'].qa.verdict.speed.score == 50


def test_speed_uses_elapsed_time_without_rewarding_more_output_or_turns():
    """Token volume and turn count cannot improve a fixed elapsed-time score."""
    entries = [entry('a', rate=10, seconds=10), entry('b', rate=200, seconds=20)]
    score_comparison(entries)
    assert entries[0]['run'].qa.verdict.speed.score == 84.1
    assert entries[1]['run'].qa.verdict.speed.score == 15.9
    entries[1]['performance']['tokens_per_second'] = 100000
    entries[1]['performance']['model_seconds_per_turn'] = .001
    score_comparison(entries)
    assert entries[1]['run'].qa.verdict.speed.score == 15.9
    score_comparison(entries[:1])
    assert entries[0]['run'].qa.verdict.speed.score is None
    assert entries[0]['run'].qa.verdict.cost.score is None


def test_saved_individual_report_explains_same_comparison_scale(tmp_path):
    """A saved copied run must export the new scale rather than old rubric anchors."""
    from reporting.render import render
    from reporting.schema import QARubric

    entries = [entry('a'), entry('b', cost=2)]
    entries[0]['run'].qa.rubric = QARubric(
        goal='Test',
        goal_achievement=[{'id': 'goal', 'description': 'Goal', 'points': 100, 'met_when': 'Met'}],
        answer_quality=[{'id': 'quality', 'description': 'Quality', 'points': 100, 'met_when': 'Met'}],
        speed={'tokens_per_second': 123, 'seconds_per_turn': 456, 'rationale': 'Old speed anchor'},
        cost={'usd_at_half_score': .0123, 'rationale': 'Old cost anchor'},
    )
    score_comparison(entries)
    run = Run.model_validate_json(entries[0]['run'].model_dump_json())
    output = tmp_path / 'report.html'
    render(run, entries[0]['prices'], output)
    html = output.read_text()
    assert 'normal-curve percentile' in html and 'population standard deviation' in html
    assert 'Old speed anchor' not in html and 'Old cost anchor' not in html
    assert '35% weight' in html and '15% weight' in html
    assert f'Overall: {run.qa.overall_score}/100' in html


def test_missing_cost_or_token_usage_does_not_erase_elapsed_speed():
    """Measured response time is independent of missing billing receipts."""
    entries = [entry('a'), entry('b')]
    entries[1]['summary']['unpriced_calls'] = 1
    entries[1]['performance']['tokens_per_second'] = None
    score_comparison(entries)
    qa = entries[1]['run'].qa
    assert qa.verdict.speed.score == 50
    assert qa.verdict.cost.score is None
    assert qa.coverage == pytest.approx(.6)
