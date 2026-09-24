"""Price normalized model usage once, independently of any report format.

Keep Decimal amounts unrounded here: rounding tiny per-call charges before
multiplying by an execution forecast would erase real costs.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from reporting.exchange import ExchangeRate
from reporting.schema import Record, Run, Step


class Rate(Record):
    """A model tariff in USD per million tokens, with verification provenance.

    For example, Rate(input="2", output="10") prices standard-rate input/output;
    omitted cache rates remain unknown, rather than implying free caching.
    as_of is the verification day, not necessarily a price's effective date.
    """

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


class ModelCalibration(Record):
    """Keep explicit calibration evidence separate from automatically refreshed prices.

    A failed check has no capacity: it must not revive a guessed denominator.
    Metadata verification establishes published capacity, not an empirical limit.
    """

    checked_at: datetime
    capacity: int | None = Field(default=None, gt=0)
    source: str | None = None
    resolved_model: str | None = None
    error: str | None = None


class Prices(Record):
    """A report's reproducible tariff and FX snapshot, including lookup failures.

    models uses exact provider:model keys; aliases are explicit mappings, never
    fuzzy matches. For example, load_prices(path) supplies this object to both
    renderers so an export does not silently substitute newer tariffs.
    """

    calibrations: dict[str, ModelCalibration] = Field(default_factory=dict)
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
    """Restore the tariff basis of an earlier run so exports remain reproducible.

    ``path`` identifies the JSON snapshot; return its validated Prices record
    for shared accounting and rendering, without looking up newer prices.

    Missing files, invalid JSON, and invalid tariffs propagate to the caller;
    silently replacing a requested file would undermine reproducible estimates.
    """
    return Prices.model_validate(json.loads(path.read_text(encoding="utf-8")))


# Ordering is shared by accounting projections and both exporters. These billed
# buckets are disjoint. The chart can regroup cache-written input as a standard
# fresh-input charge plus its rate premium without changing the total.
CATEGORIES = [
    ("input", "Input at standard rate"),
    ("cache_read", "Cache read"),
    ("cache_write_5m", "Cache write (5 min)"),
    ("cache_write_1h", "Cache write (1 hour)"),
    ("cache_write", "Cache write"),
    ("output", "Output · non-reasoning"),
    ("reasoning", "Reasoning"),
]


def breakdown(step: Step, prices: Prices) -> list[dict]:
    """Explain what each token category contributes to a recorded call's price.

    Return ordered category dictionaries for the accounting and export layers,
    keeping unknown amounts explicit so a partial estimate stays recognizable.

    Callers supply a normalized step and a saved price snapshot. A missing count
    or applicable rate produces None; a known zero count costs zero when the
    model tariff exists. Reasoning uses the output rate without charging those
    tokens again as visible output. No I/O or rounding occurs here.
    """
    key = f"{step.provider}:{step.model}"
    rate = prices.models.get(prices.aliases.get(key, key))
    usage = step.usage
    # Without usage, leave every category unknown. With usage, subtract included
    # cache/reasoning buckets so each reported token is charged exactly once.
    counts = (
        {}
        if usage is None
        else {
            "input": usage.input_tokens - usage.cache_read - usage.cache_write,
            "cache_read": usage.cache_read,
            "cache_write_5m": usage.cache_write_5m,
            "cache_write_1h": usage.cache_write_1h,
            "cache_write": usage.cache_write
            - usage.cache_write_5m
            - usage.cache_write_1h,
            "output": usage.output_tokens - usage.reasoning,
            "reasoning": usage.reasoning,
        }
    )
    result = []
    for category, label in CATEGORIES:
        count = counts.get(category)
        # Reasoning is billed at the output tariff; other buckets use their own
        # tariff. A missing model leaves all tariffs unknown, never implicitly free.
        price = (
            getattr(rate, "output" if category == "reasoning" else category)
            if rate
            else None
        )
        # The app fixes unspecified cache lifetime at five minutes; preserve an
        # explicit provider cache-write rate when one is supplied.
        if category == "cache_write" and rate and price is None:
            price = rate.cache_write_5m
        # Multiply only when both operands are known. A reported zero bucket
        # needs no category tariff, but still requires a recognized model; every
        # other missing operand must remain None for incomplete-cost reporting.
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
    """Determine whether a recorded call has enough evidence for a full price.

    ``step`` is a captured span and ``prices`` is its selected tariff snapshot.
    Return (USD amount, None) when fully priced, or (None, reason) when unknown.

    Non-model spans return (None, None): tool execution is not itself a token
    charge. A tool request's JSON is charged in the model response that emitted
    it, and tool results are charged when a later model request reads them.
    """
    # Workflow/tool spans organize execution; charging them would double-count
    # the model calls that generated and consumed their messages.
    if step.kind != "model":
        return None, None
    # A completed call without usage is unmeasured, not a zero-token call.
    if step.usage is None:
        return None, "Token usage was not reported"
    key = f"{step.provider}:{step.model}"
    # Only explicit aliases may resolve tariffs; guessing a similar model would
    # hide a configuration gap behind a plausible monetary estimate.
    if prices.aliases.get(key, key) not in prices.models:
        return None, f"No pricing entry for {key}"
    category_costs = breakdown(step, prices)
    # Any unpriced category makes the whole-call total incomplete, even when
    # the other categories have usable subtotals.
    if any(p["usd"] is None for p in category_costs):
        return None, "Cache pricing is not configured for this usage"
    return sum((p["usd"] for p in category_costs), Decimal(0)), None


def summarize(run: Run, prices: Prices) -> dict:
    """Give report readers a run-level estimate and identify gaps in that estimate.

    ``run`` supplies recorded spans; ``prices`` supplies the tariff snapshot.
    Return summary fields for export, including per-span costs keyed by span ID.

    known_cost is a subtotal of priced categories, not a promise that the entire
    run was priced. Consumers must show unpriced_calls/missing_usage alongside
    it. Parent workflow spans are excluded so nesting cannot multiply charges.
    Duration measures wall-clock span coverage, not summed nested durations.
    """
    # Select billable calls once, excluding enclosing spans. Below, omit unknown
    # amounts only from the known subtotal and unknown usage only from token sums;
    # the accompanying missing/unpriced counters preserve those omissions.
    model_steps = [s for s in run.steps if s.kind == "model"]
    costs = {s.id: cost(s, prices) for s in run.steps}
    known_cost = sum(
        (
            part["usd"]
            for model in model_steps
            for part in breakdown(model, prices)
            if part["usd"] is not None
        ),
        Decimal(0),
    )
    # An empty trace has no timestamp extrema; its duration is zero, while
    # nonempty traces use outer wall-clock bounds rather than summed spans.
    return {
        "model_calls": len(model_steps),
        "input_tokens": sum(s.usage.input_tokens for s in model_steps if s.usage),
        "output_tokens": sum(s.usage.output_tokens for s in model_steps if s.usage),
        "missing_usage": sum(s.usage is None for s in model_steps),
        "unpriced_calls": sum(costs[s.id][0] is None for s in model_steps),
        "known_cost": known_cost,
        "costs": costs,
        "duration_ms": (
            max(s.end_ns for s in run.steps) - min(s.start_ns for s in run.steps)
        )
        / 1_000_000
        if run.steps
        else 0,
    }
