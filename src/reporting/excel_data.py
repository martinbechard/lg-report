"""Convert a saved run and its pricing snapshot into data for the Excel exporter.

Reuse the HTML report's conversation order, request labels, tree, and accounting
so the spreadsheet does not invent a second interpretation of a trace. The CLI
writes a JSON projection consumed by reporting.export_excel; it does not invoke
an agent or fetch prices. Preserve unrounded costs and unknown-data flags here;
the workbook applies execution-count scaling before rounding final estimates.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
from pathlib import Path

from reporting.pricing import load_prices, summarize
from reporting.render import conversation_turns, tree_rows
from reporting.schema import Run


def workbook_data(run, prices):
    """Give the workbook writer the same interpretation of a run as the HTML report.

    ``run`` is a normalized recording and ``prices`` its saved tariff/FX record.
    Return an interchange dictionary with event details, tree rows, and a
    known-cost subtotal used by the workbook's accounting comparison.

    Reuse HTML's request numbering, event ordering, and accounting instead of
    implementing another interpretation of the trace. Return raw precision and
    missing-data flags; workbook projections round only after execution scaling.
    This function does not fetch rates or write files.
    """
    turns = conversation_turns(run, prices)
    events = []
    for turn in turns:
        for event in turn["events"]:
            step = event["step"]
            # Attach the original user prompt only to the turn's first event
            # when that event is a model request; later calls consume conversation
            # history rather than starting another user turn.
            events.append(
                {
                    "step": step.model_dump(mode="json"),
                    "turn": turn["number"],
                    "label": event.get("request_label", step.name),
                    "user_prompt": turn["request"]
                    if step.kind == "model" and event == turn["events"][0]
                    else None,
                    "cells": event["cells"],
                    "total": event["total"],
                    "partial": event["partial"],
                    "thinking": event["thinking_text"],
                    "growth": event.get("growth"),
                    "input_parts": event.get("input_parts", []),
                    "user_tokens": turn.get("request_tokens"),
                }
            )
    tree = [
        {
            "step": row["step"].model_dump(mode="json"),
            "depth": row["depth"],
            "description": row["description"],
        }
        for row in tree_rows(run, prices)
    ]
    return {
        "run": {"id": run.id, "title": run.title, "demo": run.demo},
        "events": events,
        "tree": tree,
        "prices": prices.model_dump(mode="json"),
        "expected_usd": summarize(run, prices)["known_cost"],
    }


def main():
    """Write the Excel interchange JSON for a saved run (CLI entry point).

    The adjacent prices.json belongs to that recording; fetching today's prices
    would make the workbook disagree with its HTML counterpart. Decimal values
    serialize as strings to avoid losing precision at the Python/JSON boundary.
    File and validation failures propagate instead of yielding a partial export.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Pydantic restores the recorded schema before any exporter projection;
    # malformed evidence must fail rather than create a plausible workbook.
    run = Run.model_validate_json(args.run.read_text())
    prices = load_prices(args.run.with_name("prices.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(workbook_data(run, prices), default=str))


if __name__ == "__main__":
    main()
