"""Position measured Agent cost and speed on a shared comparison bell curve.

Each metric uses the mean and population standard deviation of available live
results. Normal-CDF percentiles reward lower cost and shorter total elapsed
Agent execution, including tools and retries. Goal and answer
quality retain their judge assessments. This projection makes no model calls
and never modifies input recordings; callers explicitly choose whether to save
its copied runs alongside their individual reports.
AI assistance: Codex. Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from statistics import NormalDist, fmean, pstdev

from reporting.schema import QAComparisonScoring, QANormalDistribution, QAScore


def distribution(values):
    """Exclude unknown measurements; use the entire comparison as the population."""
    measured = [value for value in values if value is not None]
    return QANormalDistribution(count=len(measured),
                                mean=fmean(measured) if measured else None,
                                stddev=pstdev(measured) if measured else None)


def percentile(value, population, *, lower_is_better=False):
    """Map position to 0–100; ties score 50 and fewer than two values cannot rank.

    Use the signed z-score in the CDF directly to avoid subtractive cancellation
    for very good low-cost observations. Missing values never become zeros.
    """
    if value is None or population.count < 2:
        return None
    if population.stddev == 0:
        return 50.0
    z = (value - population.mean) / population.stddev
    return 100 * NormalDist().cdf(-z if lower_is_better else z)


def describe(value, population, label):
    """Keep each resource reason tied to the actual measurement and population."""
    return (f"{label}: {value:.6g}; comparison mean {population.mean:.6g}, "
            f"population standard deviation {population.stddev:.6g}, n={population.count}.")


def score_comparison(entries):
    """Replace only copied QA resource scores using every available comparison run.

    Unassessed runs may supply measurements but receive no invented judgment.
    Static runs never enter these populations. Cost requires complete pricing;
    speed uses complete outer Agent turn spans already projected by reporting.
    The saved judge overhead and simulated-user work are excluded by that
    projection. Repeated rendering always recomputes from measurements, making
    scoring independent of previous scores and the input order.
    """
    costs = [float(entry['summary']['known_cost'])
             if not entry['run'].demo and entry['summary']['model_calls']
             and not entry['summary']['unpriced_calls'] else None for entry in entries]
    times = [entry['performance']['agent_elapsed_seconds'] for entry in entries]
    basis = QAComparisonScoring(
        method='normal-elapsed-v2',
        members=[f"{', '.join(entry['models']) or entry['run'].title} · {entry['run'].id}"
                 for entry in entries],
        cost=distribution(costs), elapsed_seconds=distribution(times),
    )
    for entry, cost, seconds in zip(entries, costs, times):
        original = entry['run']
        if original.demo or not original.qa or original.qa.status != 'completed' or not original.qa.verdict:
            continue
        run = original.model_copy(deep=True)
        qa = run.qa
        cost_score = percentile(cost, basis.cost, lower_is_better=True)
        time_score = percentile(seconds, basis.elapsed_seconds, lower_is_better=True)
        qa.verdict.cost = QAScore(
            score=round(cost_score, 1) if cost_score is not None else None,
            reason=(describe(cost, basis.cost, 'Agent cost (USD)') +
                    ' Score = 100 × Φ((mean − cost) / standard deviation); equal costs score 50.'
                    if cost_score is not None else
                    'Cost unscored: requires complete Agent cost and at least two measured live results.'),
        )
        qa.verdict.speed = QAScore(
            score=round(time_score, 1) if time_score is not None else None,
            reason=(describe(seconds, basis.elapsed_seconds, 'Total Agent elapsed seconds') +
                    ' Score = 100 × Φ((mean time − elapsed time) / standard deviation). '
                    'Includes tools, tests, retries, and orchestration; excludes user waits and QA judging. '
                    'Overlapping work counts once; equal times score 50.'
                    if time_score is not None else
                    'Speed unscored: requires complete Agent turn boundaries and at least two measured live results.'),
        )
        qa.comparison_scoring = basis
        qa.speed_method = 'comparison-elapsed-v2'
        qa.calculate_overall()
        entry['run'] = run
    return basis
