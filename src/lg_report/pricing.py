"""Exact model lookup and Decimal pricing; unknown rates never imply free usage."""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from .exchange import ExchangeRate
from .schema import Record, Run, Step


class Rate(Record):
    fetched_at: datetime | None = None
    based_on: str | None = None
    as_of: date | None = None
    source: str | None = None
    input: Decimal = Field(ge=0, allow_inf_nan=False)
    output: Decimal = Field(ge=0, allow_inf_nan=False)
    cache_read: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write_5m: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write_1h: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)


class Prices(Record):
    refresh_errors: dict[str, str] = Field(default_factory=dict)
    currency: str = "USD"
    exchange: ExchangeRate | None = None
    exchange_error: str | None = None
    as_of: date
    note: str
    sources: list[str] = Field(default_factory=list)
    models: dict[str, Rate]
    aliases: dict[str, str] = Field(default_factory=dict)


def load_prices(path: Path) -> Prices:
    return Prices.model_validate(json.loads(path.read_text(encoding="utf-8")))


CATEGORIES = [
    ("input", "Fresh input"),
    ("cache_read", "Cache read"),
    ("cache_write_5m", "Cache write (5 min)"),
    ("cache_write_1h", "Cache write (1 hour)"),
    ("cache_write", "Cache write"),
    ("output", "Output · non-reasoning"),
    ("reasoning", "Reasoning"),
]


def breakdown(step: Step, prices: Prices) -> list[dict]:
    key = f"{step.provider}:{step.model}"
    rate = prices.models.get(prices.aliases.get(key, key))
    u = step.usage
    counts = (
        {}
        if u is None
        else {
            "input": u.input_tokens - u.cache_read - u.cache_write,
            "cache_read": u.cache_read,
            "cache_write_5m": u.cache_write_5m,
            "cache_write_1h": u.cache_write_1h,
            "cache_write": u.cache_write - u.cache_write_5m - u.cache_write_1h,
            "output": u.output_tokens - u.reasoning,
            "reasoning": u.reasoning,
        }
    )
    result = []
    for category, label in CATEGORIES:
        count = counts.get(category)
        price = (
            getattr(rate, "output" if category == "reasoning" else category)
            if rate
            else None
        )
        if category == "cache_write" and rate and price is None:
            price = rate.cache_write_5m
        amount = (
            Decimal(count) * price / Decimal(1_000_000)
            if count is not None and price is not None
            else Decimal(0)
            if count == 0 and rate is not None
            else None
        )
        result.append({"key": category, "label": label, "tokens": count, "usd": amount})
    return result


def cost(step: Step, prices: Prices) -> tuple[Decimal | None, str | None]:
    if step.kind != "model":
        return None, None
    if step.usage is None:
        return None, "Token usage was not reported"
    key = f"{step.provider}:{step.model}"
    if prices.aliases.get(key, key) not in prices.models:
        return None, f"No pricing entry for {key}"
    parts = breakdown(step, prices)
    if any(p["usd"] is None for p in parts):
        return None, "Cache pricing is not configured for this usage"
    return sum((p["usd"] for p in parts), Decimal(0)), None


def summarize(run: Run, prices: Prices) -> dict:
    models = [s for s in run.steps if s.kind == "model"]
    costs = {s.id: cost(s, prices) for s in run.steps}
    known_cost = sum(
        (
            part["usd"]
            for model in models
            for part in breakdown(model, prices)
            if part["usd"] is not None
        ),
        Decimal(0),
    )
    return {
        "model_calls": len(models),
        "input_tokens": sum(s.usage.input_tokens for s in models if s.usage),
        "output_tokens": sum(s.usage.output_tokens for s in models if s.usage),
        "missing_usage": sum(s.usage is None for s in models),
        "unpriced_calls": sum(costs[s.id][0] is None for s in models),
        "known_cost": known_cost,
        "costs": costs,
        "duration_ms": (
            max(s.end_ns for s in run.steps) - min(s.start_ns for s in run.steps)
        )
        / 1_000_000
        if run.steps
        else 0,
    }
