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

    @model_validator(mode="after")
    def check_time(self):
        """Reject negative elapsed time rather than presenting a plausible duration."""
        # A reversed timestamp pair indicates corrupt evidence, not zero latency.
        if self.end_ns < self.start_ns:
            raise ValueError("Step ends before it starts")
        return self

    @property
    def duration_ms(self) -> float:
        """Expose elapsed milliseconds without discarding stored timestamp precision."""
        return (self.end_ns - self.start_ns) / 1_000_000


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
