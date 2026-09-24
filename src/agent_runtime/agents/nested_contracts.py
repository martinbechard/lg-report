"""Define the structured inputs and results used by the development role agents.

These data contracts can be reused independently of any workflow. They validate
agent responses and separate public candidate source from private working notes;
workflow modules retain ownership of dispatch, context retention, and loop limits.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

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
