"""Compare independent saved runs without changing execution or billed evidence.

Each entry retains its own prices and FX snapshot. The existing pricing summary
owns arithmetic; this projection adds only comparison context and presentation.
Runs can contain several models, so totals always describe a whole run rather
than attributing delegated work to a single model. No provider calls occur here.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
import re
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote

from jinja2 import DictLoader, Environment, select_autoescape

from reporting.comparison_scoring import score_comparison
from reporting.native_timing import timing_rows
from reporting.performance import assistant_performance
from reporting.pricing import CATEGORIES, Prices, breakdown, load_prices, summarize
from reporting.render import conversation_turns, cost_chart, user_test_outcome
from reporting.schema import QA_WEIGHTS, Run


def response_text(messages: list[dict]) -> str:
    """Show captured text blocks readably; full structured evidence stays in the linked report.

    Tool-only and non-text responses receive an explicit placeholder, not an
    invented answer. Provider metadata is preserved in the linked run evidence.
    """
    text = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content:
            text.append(content)
        elif isinstance(content, list):
            text.extend(block["text"] for block in content
                        if isinstance(block, dict) and isinstance(block.get("text"), str))
    return "\n\n".join(text) or "No text response captured; see the full report for structured evidence."



def cost_recap(calls: list, prices: Prices) -> list[dict]:
    """Explain the same disjoint charges used by the chart, grouped by rate.

    Derive each effective rate from the canonical unrounded charge and count,
    so long-context tariffs and cache lifetime rules cannot drift here. Unknown
    usage/rates remain explicit; zero buckets need no multiplication expression.
    Different tariffs receive separate terms instead of a misleading average.
    """
    parts = [breakdown(call, prices) for call in calls]
    rows = []
    for index, (key, label) in enumerate(CATEGORIES):
        buckets = [part[index] for part in parts]
        terms = {}
        for bucket in buckets:
            count, amount = bucket["tokens"], bucket["usd"]
            if count:
                rate = amount * Decimal(1_000_000) / count if amount is not None else None
                terms[rate] = terms.get(rate, 0) + count
        rows.append({"key": key, "label": label,
                     "terms": [{"tokens": count, "rate": rate} for rate, count in terms.items()],
                     "unknown": not calls or any(b["usd"] is None for b in buckets),
                     "missing_usage": any(b["tokens"] is None for b in buckets),
                     "tokens": sum(b["tokens"] or 0 for b in buckets),
                     "usd": sum((b["usd"] for b in buckets if b["usd"] is not None), Decimal(0))})
    return rows


def comparison_entry(run: Run, prices: Prices, source: str) -> dict:
    """Project one run without merging span IDs or replacing missing measurements.

    Repeated span/run IDs across entries are valid: independent recordings do
    not share an identity namespace. Recorded requests and responses remain
    available for readers to assess workload and answer quality themselves.
    """
    if prices.currency != "USD":
        raise ValueError("Comparison requires a USD pricing table")
    if prices.exchange and (
        prices.exchange.base != "USD" or prices.exchange.quote != "EUR"
    ):
        raise ValueError("Comparison exchange rate must describe USD to EUR")
    user_calls = [step for step in run.steps
                  if step.kind == "model" and step.context.get("model_role") == "user"]
    # Keep source evidence intact. In the assistant-only projection user calls
    # become non-billable enclosing operations, preserving valid parent links
    # and elapsed run time while excluding test-input generation from the plot.
    target = run.model_copy(update={"steps": [
        step.model_copy(update={"kind": "workflow"}) if step in user_calls else step
        for step in run.steps
    ]})
    models = sorted((step for step in target.steps if step.kind == "model"),
                    key=lambda step: step.start_ns)
    summary = summarize(target, prices)
    # Price completeness and token completeness are independent. A missing
    # tariff does not erase observed tokens; missing usage prevents a full cost.
    return {
        "run": run,
        "prices": prices,
        "source": source,
        "summary": summary,
        "performance": assistant_performance(run),
        "cost_recap": cost_recap(models, prices),
        "native_timing": timing_rows(models) if not run.demo else [],
        "execution_summary": summarize(run, prices),
        "models": sorted({
            f"{step.provider or 'Unknown provider'}:{step.model or 'Unknown model'}"
            for step in models
        }),
        "efforts": sorted({step.effort or "Not recorded" for step in models}),
        "calls": models,
        "chart": cost_chart(conversation_turns(target, prices), prices),
        "user_models": sorted({f"{step.provider}:{step.model}" for step in user_calls}),
        "user_test": user_test_outcome(run),
        "user_summary": summarize(run.model_copy(update={"steps": user_calls}), prices),
    }


def comparison_rows(entries: list[dict]) -> tuple[list[dict], bool]:
    """Align recorded turns, preserving absent responses and unequal prompts.

    Without explicit turn metadata, request order is the only defensible
    alignment. Multiple calls in a turn remain labeled responses, not an
    invented final answer from an arbitrary specialist.
    """
    by_turn = all(type(call.context.get("report_turn")) is int
                  for entry in entries for call in entry["calls"])
    groups = []
    for entry in entries:
        grouped = {}
        for index, call in enumerate(entry["calls"], 1):
            key = call.context["report_turn"] if by_turn else index
            grouped.setdefault(key, []).append(call)
        groups.append(grouped)
    rows = []
    for key in sorted({key for group in groups for key in group}):
        cells = []
        for group in groups:
            calls = group.get(key, [])
            # The latest human message is the current prompt, not earlier
            # conversation history. Missing capture is never a matching prompt.
            prompt = next((response_text([message]) for message in reversed(calls[0].request)
                           if message.get("role") in {"human", "user"}), None) if calls else None
            cells.append({"calls": calls, "prompt": prompt})
        prompts = [cell["prompt"] for cell in cells]
        shared = prompts[0] if prompts[0] and all(p == prompts[0] for p in prompts) else None
        rows.append({"label": f"{'Turn' if by_turn else 'Request'} {key}",
                     "prompt": shared, "cells": cells})
    return rows, by_turn


def comparison_charts(entries: list[dict]) -> list[dict]:
    """Plot every saved run together on one EUR and context-token diagram.

    cost_chart owns cost stacks, cache-write premiums, retained-output context,
    history identity, and missing-data flags. Only SVG placement changes here.
    Requests share an ordinal position so line spacing is independent of model count.
    Model colors identify solid cumulative-cost and dashed context lines.
    """
    charts = [entry["chart"] for entry in entries if entry["chart"]]
    cost_max = max((bar["cumulative"] for chart in charts for bar in chart["bars"]),
                   default=Decimal(0)) * Decimal("1.15") or Decimal(1)
    token_max = max((chart["token_max"] for chart in charts), default=1)
    count = max((len(chart["bars"]) for chart in charts), default=1)
    # Keep the axis margins while halving the space between request positions.
    width = max(600, 200 + count * 30)
    palette = ["#2458a6", "#9c3f75", "#17734b", "#b35c16", "#6445a3", "#007f8b"]
    for index, entry in enumerate(entries):
        entry["color"] = palette[index % len(palette)]
        entry["series_number"] = index + 1
    margin = 100
    positions = [width / 2 if count == 1 else margin + i * (width - 2 * margin) / (count - 1)
                 for i in range(count)]
    legend = {}
    for entry in entries:
        chart = entry["chart"]
        if chart is None:
            continue
        for item in chart["legend"]:
            legend[item["label"]] = item
        chart["comparison_width"] = width
        chart["request_groups"] = [{"x": x, "number": i + 1} for i, x in enumerate(positions)]
        chart["comparison_ticks"] = [
            {"y": 335 - i * 70, "cost": cost_max * i / 4,
             "tokens": round(token_max * i / 4)} for i in range(5)
        ]
        previous = {}
        context_lines = []
        for index, bar in enumerate(chart["bars"]):
            x = positions[index]
            bar["comparison_x"] = x
            bar["comparison_cost_y"] = 335 - float(bar["cumulative"] / cost_max) * 280
            bar["comparison_context_y"] = (335 - bar["context_tokens"] / token_max * 280
                                            if bar["context_tokens"] is not None else None)
            # Never bridge unknown context or connect independent histories.
            if bar["summary_request"]:
                continue
            history = bar["history_id"]
            point = (x, bar["comparison_context_y"])
            if previous.get(history) and point[1] is not None:
                context_lines.append({"start": previous[history], "end": point,
                                      "color": entry["color"]})
            previous[history] = point if point[1] is not None else None
        chart["comparison_context_lines"] = context_lines
        chart["comparison_cost_line"] = " ".join(
            f"{bar['comparison_x']},{bar['comparison_cost_y']}" for bar in chart["bars"]
        )
    return list(legend.values())


def render_comparison(
    run_paths: list[Path], destination: Path, *, title: str | None = None
) -> None:
    """Write standalone HTML from at least two run.json/prices.json pairs.

    Preserve adjacent saved tariffs and FX even when two runs used different
    snapshots. Relative source names make the supporting bundles discoverable
    without embedding the local user's absolute filesystem paths. Validate all
    inputs before writing; the destination's parent must exist, as for render.
    """
    if len(run_paths) < 2:
        raise ValueError("Comparison requires at least two saved runs")
    # A typo must never replace input evidence with the generated presentation.
    inputs = {path.resolve() for path in run_paths}
    inputs.update(path.with_name("prices.json").resolve() for path in run_paths)
    if destination.resolve() in inputs:
        raise ValueError("Comparison output must not overwrite run or price evidence")
    entries = []
    for path in run_paths:
        run = Run.model_validate_json(path.read_text(encoding="utf-8"))
        prices = load_prices(path.with_name("prices.json"))
        entry = comparison_entry(run, prices, os.path.relpath(path.resolve(), destination.parent.resolve()))
        report = path.with_name("report.html")
        # Quote URL syntax in filenames; never produce a broken full-report link
        # when a bundle only contains JSON evidence.
        entry["report_href"] = quote(os.path.relpath(report.resolve(), destination.parent.resolve())) if report.is_file() else None
        entry["source_href"] = quote(entry["source"])
        # Show one shared judge above the table. Preserve per-run identity only
        # when provider, model, or recorded reasoning effort actually differs.
        qa = run.qa
        efforts = sorted({step.effort for step in qa.judge_steps if step.effort}) if qa else []
        entry["qa_judge"] = (
            f"{qa.provider}:{qa.model}" + (f" · {', '.join(efforts)} effort" if efforts else "")
            if qa else None
        )
        entries.append(entry)
    # Shared application names come from saved run metadata, just as in the
    # individual report. An explicit title remains available for mixed applications.
    if title is None:
        titles = {entry["run"].title for entry in entries}
        title = f"Application: {next(iter(titles))}" if len(titles) == 1 else "Model comparison"
    comparison_scoring = score_comparison(entries)
    qa_judges = list(dict.fromkeys(entry["qa_judge"] for entry in entries if entry["qa_judge"]))
    # Equality includes every check, weight and numerical anchor. A shared judge
    # name alone cannot make independently invented criteria comparable.
    rubrics = [entry["run"].qa.rubric if entry["run"].qa else None for entry in entries]
    shared_rubric = (rubrics[0] if all(rubrics) and
                     len({rubric.fingerprint for rubric in rubrics}) == 1 else None)
    mixed_rubrics = any(rubrics) and shared_rubric is None
    env = Environment(
        autoescape=select_autoescape(default=True),
        loader=DictLoader({name: files("reporting").joinpath(f"templates/{name}").read_text(encoding="utf-8")
                           for name in ("comparison_qa.html", "qa_criteria.html")}),
    )
    env.filters["response_text"] = response_text
    rows, by_turn = comparison_rows(entries)
    legend = comparison_charts(entries)
    shared_chart = next((entry["chart"] for entry in entries
                         if entry["chart"] and entry["chart"]["bars"]), None)
    template = env.from_string(
        files("reporting").joinpath("templates/comparison.html").read_text(encoding="utf-8")
    )
    html = template.render(title=title, entries=entries, rows=rows, by_turn=by_turn, qa_weights=QA_WEIGHTS,
                           legend=legend, shared_chart=shared_chart, qa_judges=qa_judges,
                           shared_rubric=shared_rubric, mixed_rubrics=mixed_rubrics,
                           comparison_scoring=comparison_scoring)
    # Preserve captured trailing spaces in the rendered text while keeping the
    # generated HTML free of source whitespace errors (common Markdown breaks).
    html = re.sub(r" +(?=\n)", lambda match: "&#32;" * len(match[0]), html)
    destination.write_text(html, encoding="utf-8")
