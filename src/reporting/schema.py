"""Define the provider-independent trace contract shared by report exporters.

AI attribution: Generated with AI assistance.

Keep observed usage separate from prices so the same run.json can be repriced or
exported to HTML and Excel without re-running an agent or changing its evidence.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    """Reject unknown fields so schema drift fails visibly at file boundaries."""

    model_config = ConfigDict(extra="forbid")


class Usage(Record):
    """Store inclusive request/response totals with their billing subsets.

    For example, Usage(input_tokens=100, output_tokens=20, cache_read=80)
    represents 20 fresh input tokens, not 180 input tokens. Cache lifetimes split
    cache_write; reasoning splits output_tokens. A missing Usage on Step means
    usage is unknown, whereas zero within an existing Usage is a recorded value.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)
    cache_write_5m: int = Field(default=0, ge=0)
    cache_write_1h: int = Field(default=0, ge=0)
    reasoning: int = Field(default=0, ge=0)

    # Pydantic calls this after parsing fields so all exporters can rely on
    # internally consistent token accounting. self is the parsed Usage; return
    # it unchanged on success, or raise to reject the whole invalid record.
    @model_validator(mode="after")
    def check_subsets(self):
        """Reject impossible subset totals before exporters calculate charges."""
        # Lifetime buckets partition the write total; exceeding it would price
        # more cache creation than the provider reported.
        if self.cache_write_5m + self.cache_write_1h > self.cache_write:
            raise ValueError("Cache lifetime details must be a subset of cache writes")
        # Reads and writes are disjoint subsets of inclusive input. An excess
        # would make fresh input negative and invalidate every cost breakdown.
        if self.cache_read + self.cache_write > self.input_tokens:
            raise ValueError("Cache tokens must be a subset of input tokens")
        # Reasoning is included in output, not an additional output total; an
        # excess would imply a negative visible-response token count.
        if self.reasoning > self.output_tokens:
            raise ValueError("Reasoning tokens must be a subset of output tokens")
        return self


class Step(Record):
    """Preserve one observed operation and its ancestry, independent of display.

    Times are Unix nanoseconds; parent_id links callback operations, including
    subagent work. Usage belongs to the model call that incurred it; parent
    subtree totals are derived later, not stored as additional billed usage.
    Request/response content can be empty because capture was disabled.
    """

    id: str
    parent_id: str | None = None
    name: str
    kind: Literal["workflow", "model", "tool", "retriever"]
    start_ns: int
    end_ns: int
    status: Literal["ok", "error", "interrupted", "incomplete"]
    provider: str | None = None
    model: str | None = None
    effort: str | None = None
    usage: Usage | None = None
    error: str | None = None
    request: list[dict] = Field(default_factory=list)
    response: list[dict] = Field(default_factory=list)
    # None means not captured; an empty list means explicitly no bound tools.
    tool_definitions: list[dict] | None = None
    context: dict[str, str | int | list[str]] = Field(default_factory=dict)

    # Pydantic calls this on the parsed Step to keep latency calculations
    # meaningful. A valid instance is returned unchanged; reversed times reject
    # the record instead of being silently corrected.
    @model_validator(mode="after")
    def check_time(self):
        """Reject negative elapsed time rather than presenting a plausible duration."""
        # A reversed timestamp pair indicates corrupt evidence, not zero latency.
        if self.end_ns < self.start_ns:
            raise ValueError("Step ends before it starts")
        return self

    # Renderers read this property for a human-scale duration. It derives a
    # float from this Step's timestamps without replacing the original evidence.
    @property
    def duration_ms(self) -> float:
        """Expose elapsed milliseconds without discarding stored timestamp precision."""
        return (self.end_ns - self.start_ns) / 1_000_000


class QAScore(Record):
    """Keep an evidence-based score separate from a missing assessment."""

    score: float | None = Field(ge=0, le=100, allow_inf_nan=False, strict=True)
    reason: str = Field(min_length=1)


class QAVerdict(Record):
    """Require all four rubric dimensions; null means insufficient evidence."""

    goal_achievement: QAScore
    answer_quality: QAScore
    speed: QAScore
    cost: QAScore
    summary: str = Field(min_length=1)


class QACriterion(Record):
    """Define an observable check and its share of a dimension's 100 points."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    points: int = Field(gt=0, le=100, strict=True)
    met_when: str = Field(min_length=1)
    # Goal checks are binary. Quality checks may award half credit only under
    # this predeclared condition, never through an improvised numerical score.
    partial_when: str | None = None


class QASpeedScale(Record):
    """Fix throughput and latency half-credit anchors before viewing results."""

    tokens_per_second: float = Field(gt=0, allow_inf_nan=False)
    seconds_per_turn: float = Field(gt=0, allow_inf_nan=False)
    rationale: str = Field(min_length=1)


class QACostScale(Record):
    """Fix the total Agent cost earning 50 points, independent of competitors."""

    usd_at_half_score: float = Field(gt=0, allow_inf_nan=False)
    rationale: str = Field(min_length=1)


class QARubric(Record):
    """Save one task-specific scoring contract shared by every evaluation.

    Criteria are authored or generated from the task before results are judged. The
    fingerprint compares the full contract, including the goal and all anchors.
    """

    goal: str = Field(min_length=1)
    goal_achievement: list[QACriterion] = Field(min_length=1, max_length=12)
    answer_quality: list[QACriterion] = Field(min_length=1, max_length=12)
    speed: QASpeedScale
    cost: QACostScale

    @model_validator(mode="after")
    def check_criteria(self):
        """Reject ambiguous IDs, inconsistent totals, and partial goal credit."""
        criteria = self.goal_achievement + self.answer_quality
        if len({c.id for c in criteria}) != len(criteria):
            raise ValueError("Rubric criterion IDs must be unique")
        for dimension in (self.goal_achievement, self.answer_quality):
            if sum(c.points for c in dimension) != 100:
                raise ValueError("Each rubric dimension must total 100 points")
        if any(c.partial_when is not None for c in self.goal_achievement):
            raise ValueError("Goal criteria must be binary")
        return self

    @property
    def fingerprint(self) -> str:
        """Identify identical contracts without trusting a model-authored label."""
        import hashlib
        import json

        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()


class QACheck(Record):
    """Record the judge's evidence decision; the application assigns points."""

    criterion_id: str
    outcome: Literal["met", "partial", "unmet", "unknown"]
    evidence_ids: list[str]
    reason: str = Field(min_length=1)


class QAAssessment(Record):
    """Allow evidence classifications, never arbitrary numeric scores."""

    goal_achievement: list[QACheck]
    answer_quality: list[QACheck]
    summary: str = Field(min_length=1)


# Fixed weights make scores comparable and prevent the judge changing arithmetic.
QA_WEIGHTS = {"goal_achievement": 0.35, "answer_quality": 0.15, "speed": 0.10, "cost": 0.40}


class QANormalDistribution(Record):
    """Save a comparison population's parameters so percentiles are auditable."""

    count: int = Field(ge=0)
    mean: float | None = None
    stddev: float | None = Field(default=None, ge=0)


class QAComparisonScoring(Record):
    """Identify the cohort behind resource scores without changing judge criteria."""

    method: Literal["normal-percentile-v1", "normal-elapsed-v2"] = "normal-percentile-v1"
    members: list[str]
    cost: QANormalDistribution
    # Preserve historical throughput-based populations when loading old runs.
    output_rate: QANormalDistribution | None = None
    seconds_per_turn: QANormalDistribution | None = None
    elapsed_seconds: QANormalDistribution | None = None


class QAEvaluation(Record):
    """Persist independent QA and its overhead without changing workflow totals.

    Score coverage is the assessed fraction of the fixed rubric. A partial
    overall score uses only assessed dimensions and must display its coverage.
    Judge steps contain metadata/usage only, never a second copy of the prompt.
    """

    status: Literal["completed", "error", "skipped"]
    provider: str
    model: str
    rubric_version: Literal[1, 2] = 1
    rubric: QARubric | None = None
    assessment: QAAssessment | None = None
    # Legacy saved judgments retain their original basis on load.
    speed_method: Literal["judge-v1", "measured-v1", "shared-rubric-v2", "comparison-normal-v1", "comparison-elapsed-v2"] = "judge-v1"
    # Relative resource scores retain their population, including standalone
    # exports of a compared run. The original judge rubric stays unchanged.
    comparison_scoring: QAComparisonScoring | None = None
    # Older saved assessments included test-input generation in resource use.
    # Retain that distinction rather than silently relabeling historical scores.
    measurement_scope: Literal["execution", "assistant"] = "execution"
    goal: str | None = None
    execution_duration_ms: float | None = None
    execution_cost_usd: str | None = None
    verdict: QAVerdict | None = None
    error: str | None = None
    truncated: bool = False
    judge_steps: list[Step] = Field(default_factory=list)
    duration_ms: float = 0
    cost_usd: str | None = None

    coverage: float = 0
    overall_score: float | None = None

    @model_validator(mode="after")
    def calculate_overall(self):
        """Recompute derived values on load so stored arithmetic cannot drift."""
        self.coverage = sum(weight for key, weight in QA_WEIGHTS.items()
                            if self.verdict and getattr(self.verdict, key).score is not None)
        self.overall_score = (
            round(sum(getattr(self.verdict, key).score * weight
                      for key, weight in QA_WEIGHTS.items()
                      if getattr(self.verdict, key).score is not None) / self.coverage, 1)
            if self.coverage else None
        )
        return self


class Run(Record):
    """A serializable report input, including incomplete or failed executions.

    Multiple roots are valid: consecutive user turns can invoke the graph
    separately. Run.model_validate_json(...) validates saved evidence before any
    exporter follows parent links or computes totals.
    """

    schema_version: Literal[1] = 1
    id: str
    title: str
    demo: bool = False
    status: Literal["ok", "error", "interrupted", "incomplete"]
    steps: list[Step]
    output: str | None = None
    qa: QAEvaluation | None = None

    # Pydantic invokes this on the parsed Run before reporting can traverse its
    # steps. It establishes structural ancestry only, not semantic correctness
    # of agent actions. Return self unchanged or reject the entire invalid run.
    @model_validator(mode="after")
    def check_tree(self):
        """Require unique IDs and resolvable, acyclic ancestry for safe traversal."""
        ids = {s.id for s in self.steps}
        # Duplicate identifiers would overwrite entries in the ancestry map and
        # make costs or children attach to the wrong operation.
        if len(ids) != len(self.steps):
            raise ValueError("Duplicate step ID")
        parents = {s.id: s.parent_id for s in self.steps}
        for step in self.steps:
            seen = {step.id}
            parent = step.parent_id
            while parent is not None:
                # A missing parent is a broken reference; a previously visited
                # ancestor is a cycle (including self-parenting). Either prevents
                # a safe walk to a root, so reject instead of silently truncating.
                if parent not in ids or parent in seen:
                    raise ValueError("Invalid or cyclic step ancestry")
                seen.add(parent)
                parent = parents[parent]
        return self
