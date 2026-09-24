"""Enforce the nested-workflows sample's finite-loop and handoff policies.

This sample-specific policy is shared only by its orchestration and fixtures.
Reusable agents own their data contracts without importing workflow policy.
A review approval permits testing; only a passing test assessment permits finish.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from dataclasses import dataclass

from agent_runtime.agents.nested_contracts import Artifact, Assessment


@dataclass(frozen=True)
class Limits:
    """Count code-review rounds per visit and coding-workflow visits per task.

    The outer count is never reset when the child is re-entered after a test
    failure. Together these limits allow at most their product in coder turns.
    They are loop guards, not the timed half-open circuit-breaker pattern.
    """

    max_review_rounds: int = 3
    max_coding_cycles: int = 3

    def __post_init__(self):
        for value in (self.max_review_rounds, self.max_coding_cycles):
            if type(value) is not int or value < 1:
                raise ValueError("Loop limits must be positive integers")


def supervisor_action(review: dict | None, rounds: int, limit: int) -> str:
    """Accept a passing final round; stop a failed final round without a retry."""
    if review is not None and review["verdict"] == "pass":
        return "return"
    if rounds >= limit:
        return "escalate"
    return "code"


def planner_action(
    coding: dict | None, testing: dict | None, cycles: int, limit: int
) -> str:
    """Protect the outer test-fix loop as well as the nested review loop."""
    if coding is not None and coding["status"] == "blocked":
        return "escalate"
    if testing is not None:
        if coding is None or coding["status"] != "approved_for_testing":
            raise ValueError("Testing requires review-approved code")
        bind_assessment(testing, coding["artifact"])
        if testing["verdict"] == "pass":
            return "finish"
        return "code" if cycles < limit else "escalate"
    if coding is not None and coding["status"] == "approved_for_testing":
        return "test"
    return "code" if cycles < limit else "escalate"


def bind_assessment(assessment: dict, artifact: dict) -> None:
    """Never apply an old approval to a replacement candidate."""
    Assessment.model_validate(assessment)
    Artifact.model_validate(artifact)
    if assessment["candidate_id"] != artifact["candidate_id"]:
        raise ValueError("Assessment refers to a different candidate")


def require_action(actual: str, expected: str) -> None:
    """Fail closed rather than letting model text override a required gate."""
    if actual != expected:
        raise ValueError(f"Agent requested {actual!r}; workflow permits {expected!r}")
