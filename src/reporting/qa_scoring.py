"""Calculate QA scores from a frozen rubric and evidence classifications.

The judge selects observable outcomes; it cannot invent points, move numeric
anchors after seeing competitors, or adjust totals. Agent-only measurements
come from the same accounting and timing projections used by the reports.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from decimal import Decimal

from reporting.performance import assistant_performance
from reporting.schema import QAScore, QAVerdict


def score_assessment(rubric, assessment, run, cost_usd):
    """Require every defined check once and derive all four constituent scores."""
    event_ids = {step.id for step in run.steps}
    scores = {}
    for dimension in ("goal_achievement", "answer_quality"):
        criteria = getattr(rubric, dimension)
        checks = getattr(assessment, dimension)
        if (len(checks) != len(criteria)
                or {c.criterion_id for c in checks} != {c.id for c in criteria}):
            raise ValueError("Assessment must cover each criterion exactly once")
        by_id = {check.criterion_id: check for check in checks}
        points, reasons, unknown = 0, [], False
        for criterion in criteria:
            check = by_id[criterion.id]
            if set(check.evidence_ids) - event_ids:
                raise ValueError("Assessment cites unknown evidence")
            if check.outcome in {"met", "partial"} and not check.evidence_ids:
                raise ValueError("Awarded credit requires recorded evidence")
            if check.outcome == "partial" and not criterion.partial_when:
                raise ValueError("Partial credit is not defined for this criterion")
            earned = criterion.points * {"met": 1, "partial": .5, "unmet": 0, "unknown": 0}[check.outcome]
            points += earned
            unknown |= check.outcome == "unknown"
            label = "unscored" if check.outcome == "unknown" else f"{earned:g}/{criterion.points}"
            reasons.append(f"{criterion.description} ({label}): {check.reason}")
        scores[dimension] = QAScore(score=None if unknown else float(points), reason="\n".join(reasons))
    metrics = assistant_performance(run)
    rate, seconds = metrics["tokens_per_second"], metrics["model_seconds_per_turn"]
    if rate is None or seconds is None:
        scores["speed"] = QAScore(score=None, reason="Complete Agent timing and usage are unavailable.")
    else:
        scale = rubric.speed
        value = 50 * rate / (rate + scale.tokens_per_second) + 50 * scale.seconds_per_turn / (seconds + scale.seconds_per_turn)
        scores["speed"] = QAScore(score=round(value, 1), reason=(
            f"{rate:.2f} output tokens/s; {seconds:.2f} seconds/turn. "
            f"50 × rate/(rate + {scale.tokens_per_second:g}) + "
            f"50 × {scale.seconds_per_turn:g}/(seconds + {scale.seconds_per_turn:g})."))
    if cost_usd is None:
        scores["cost"] = QAScore(score=None, reason="Complete Agent cost is unavailable.")
    else:
        cost, anchor = Decimal(cost_usd), Decimal(str(rubric.cost.usd_at_half_score))
        scores["cost"] = QAScore(score=round(float(100 * anchor / (cost + anchor)), 1), reason=(
            f"Agent cost ${cost}; 100 × {anchor}/(cost + {anchor})."))
    return QAVerdict(**scores, summary=assessment.summary)
