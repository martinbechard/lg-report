"""Compare independent saved runs without changing execution or billed evidence.

Each entry retains its own prices and FX snapshot. The existing pricing summary
owns arithmetic; this projection adds only comparison context and presentation.
Runs can contain several models, so totals always describe a whole run rather
than attributing delegated work to a single model. No provider calls occur here.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, select_autoescape

from reporting.pricing import Prices, load_prices, summarize
from reporting.render import conversation_turns, cost_chart
from reporting.schema import Run


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
    models = sorted((step for step in run.steps if step.kind == "model"),
                    key=lambda step: step.start_ns)
    summary = summarize(run, prices)
    # Price completeness and token completeness are independent. A missing
    # tariff does not erase observed tokens; missing usage prevents a full cost.
    return {
        "run": run,
        "prices": prices,
        "source": source,
        "summary": summary,
        "models": sorted({
            f"{step.provider or 'Unknown provider'}:{step.model or 'Unknown model'}"
            for step in models
        }),
        "efforts": sorted({step.effort or "Not recorded" for step in models}),
        "calls": models,
        "chart": cost_chart(conversation_turns(run, prices), prices),
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
    Requests share an ordinal group; model offsets keep cost stacks distinct.
    Model colors identify solid cumulative-cost and dashed context lines.
    """
    charts = [entry["chart"] for entry in entries if entry["chart"]]
    cost_max = max((bar["cumulative"] for chart in charts for bar in chart["bars"]),
                   default=Decimal(0)) * Decimal("1.15") or Decimal(1)
    token_max = max((chart["token_max"] for chart in charts), default=1)
    count = max((len(chart["bars"]) for chart in charts), default=1)
    width = max(1000, 200 + count * max(100, len(entries) * 30 + 40))
    palette = ["#2458a6", "#9c3f75", "#17734b", "#b35c16", "#6445a3", "#007f8b"]
    for index, entry in enumerate(entries):
        entry["color"] = palette[index % len(palette)]
        entry["series_number"] = index + 1
    margin = 100 + len(entries) * 13
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
            x = positions[index] + (entry["series_number"] - (len(entries) + 1) / 2) * 26
            bar["comparison_x"] = x
            bar["comparison_cost_y"] = 335 - float(bar["cumulative"] / cost_max) * 280
            bar["comparison_context_y"] = (335 - bar["context_tokens"] / token_max * 280
                                            if bar["context_tokens"] is not None else None)
            accumulated = 0
            for segment in bar["segments"]:
                height = float(segment["eur"] / cost_max) * 280
                accumulated += height
                segment["comparison_y"] = 335 - accumulated
                segment["comparison_height"] = height
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
    run_paths: list[Path], destination: Path, *, title: str = "Model comparison"
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
        entries.append(entry)
    env = Environment(autoescape=select_autoescape(default=True))
    env.filters["response_text"] = response_text
    rows, by_turn = comparison_rows(entries)
    legend = comparison_charts(entries)
    shared_chart = next((entry["chart"] for entry in entries
                         if entry["chart"] and entry["chart"]["bars"]), None)
    template = env.from_string(
        files("reporting").joinpath("templates/comparison.html").read_text(encoding="utf-8")
    )
    destination.write_text(template.render(title=title, entries=entries, rows=rows, by_turn=by_turn, legend=legend, shared_chart=shared_chart), encoding="utf-8")
