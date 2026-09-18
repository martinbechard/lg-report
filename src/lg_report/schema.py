"""Versioned, provider-independent report data. Costs are derived separately."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Usage(Record):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)
    cache_write_5m: int = Field(default=0, ge=0)
    cache_write_1h: int = Field(default=0, ge=0)
    reasoning: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_subsets(self):
        if self.cache_write_5m + self.cache_write_1h > self.cache_write:
            raise ValueError("Cache lifetime details must be a subset of cache writes")
        if self.cache_read + self.cache_write > self.input_tokens:
            raise ValueError("Cache tokens must be a subset of input tokens")
        if self.reasoning > self.output_tokens:
            raise ValueError("Reasoning tokens must be a subset of output tokens")
        return self


class Step(Record):
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
    context: dict[str, str | int | list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_time(self):
        if self.end_ns < self.start_ns:
            raise ValueError("Step ends before it starts")
        return self

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000


class Run(Record):
    schema_version: Literal[1] = 1
    id: str
    title: str
    demo: bool = False
    status: Literal["ok", "error", "interrupted", "incomplete"]
    steps: list[Step]
    output: str | None = None

    @model_validator(mode="after")
    def check_tree(self):
        ids = {s.id for s in self.steps}
        if len(ids) != len(self.steps):
            raise ValueError("Duplicate step ID")
        parents = {s.id: s.parent_id for s in self.steps}
        for step in self.steps:
            seen = {step.id}
            parent = step.parent_id
            while parent is not None:
                if parent not in ids or parent in seen:
                    raise ValueError("Invalid or cyclic step ancestry")
                seen.add(parent)
                parent = parents[parent]
        return self
