"""Define handoff data and finite-loop policies for the nested-workflows lesson.

These pure contracts contain no messages, models, or framework state. Runtime
code, not an agent's prose, decides which transition is currently permitted.
A review approval means ready for testing, never that testing has passed.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Reject malformed or misspelled fields before they can affect routing."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Task(Contract):
    """Stable requirements projected by the planner, not its full conversation."""

    objective: str = Field(min_length=1)
    acceptance: list[str] = Field(min_length=1)
    constraints: list[str]

    @model_validator(mode="after")
    def nonblank(self):
        """A syntactically present requirement must also contain meaningful text."""
        if not self.objective.strip() or any(
            not value.strip() for value in self.acceptance + self.constraints
        ):
            raise ValueError("Task text must not be blank")
        return self


class Artifact(Contract):
    """The candidate that may cross context boundaries, without coder notes."""

    candidate_id: str = Field(min_length=1)
    source: str = Field(min_length=1)


class Implementation(Artifact):
    """A coder's response; working notes remain in the coding context only."""

    working_notes: str = Field(min_length=1)

    def handoff(self) -> Artifact:
        """Select public artifact fields rather than copying the entire response."""
        return Artifact(candidate_id=self.candidate_id, source=self.source)


class Assessment(Contract):
    """Bind a verdict to one candidate and distinguish scripted from live judgment.

    This lesson has no shell or test-execution tool. Neither permitted evidence
    kind claims that code was executed. Live mode is an LLM assessment only.
    """

    candidate_id: str = Field(min_length=1)
    verdict: Literal["pass", "fail"]
    findings: list[str]
    evidence: str = Field(min_length=1)
    evidence_kind: Literal["scripted", "model_assessment"]

    @model_validator(mode="after")
    def consistent(self):
        """Do not let a contradictory approval bypass a required repair."""
        if not self.evidence.strip():
            raise ValueError("An assessment needs an explanation")
        if any(not item.strip() for item in self.findings):
            raise ValueError("Findings must be nonblank")
        if self.verdict == "pass" and self.findings:
            raise ValueError("A pass cannot include unresolved findings")
        if self.verdict == "fail" and not self.findings:
            raise ValueError("A failure needs actionable findings")
        return self


class PlannerDecision(Contract):
    """An agent issues the permitted dispatch and maintains a stable task contract."""

    action: Literal["code", "test", "finish", "escalate"]
    task: Task
    note: str = Field(min_length=1)


class SupervisorDecision(Contract):
    """Coding instructions or a bounded return, never authority to skip a gate."""

    action: Literal["code", "return", "escalate"]
    directive: str = Field(min_length=1)


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


def planner_action(coding: dict | None, testing: dict | None,
                   cycles: int, limit: int) -> str:
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
