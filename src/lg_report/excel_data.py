"""Prepare the existing report accounting for the Excel workbook builder."""

import argparse
import json
from pathlib import Path

from .pricing import load_prices, summarize
from .render import conversation_turns, tree_rows
from .schema import Run


def workbook_data(run, prices):
    turns = conversation_turns(run, prices)
    events = []
    for turn in turns:
        for event in turn["events"]:
            step = event["step"]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = Run.model_validate_json(args.run.read_text())
    prices = load_prices(args.run.with_name("prices.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(workbook_data(run, prices), default=str))


if __name__ == "__main__":
    main()
