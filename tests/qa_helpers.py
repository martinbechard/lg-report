"""Provide deterministic shared criteria and judge evidence for offline tests.

Fixtures give quality checks unequal weights to catch incorrect arithmetic;
no provider credentials or live models are used.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from reporting.schema import QARubric


def rubric_definition(goal="Explain caching"):
    """Return a validated task contract with binary goals and anchored quality."""
    return QARubric.model_validate({
        "goal": goal,
        "goal_achievement": [{"id": "goal", "description": "Explain the concept", "points": 100,
                               "met_when": "The explanation covers the requested concept."}],
        "answer_quality": [
            {"id": "correct", "description": "Correctness", "points": 60,
             "met_when": "All material claims are correct.", "partial_when": "One material error."},
            {"id": "clear", "description": "Clarity", "points": 40,
             "met_when": "Clear explanation with a useful example.", "partial_when": "Clear without an example."}],
        "speed": {"tokens_per_second": 25, "seconds_per_turn": 5, "rationale": "Fixed task calibration."},
        "cost": {"usd_at_half_score": .000014, "rationale": "Fixed task calibration."},
    })


def assessment_text(event_id="answer"):
    """Award the full goal and 80 quality points through observed classifications."""
    def check(key, outcome):
        return {"criterion_id": key, "outcome": outcome, "evidence_ids": [event_id],
                "reason": "Observed in the recorded answer."}
    return json.dumps({"goal_achievement": [check("goal", "met")],
                       "answer_quality": [check("correct", "met"), check("clear", "partial")],
                       "summary": "The concept is explained; an example would improve clarity."})
