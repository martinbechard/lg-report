"""Self-contained expandable trace tree and disjoint token/cost columns."""

import json
from ast import literal_eval
from datetime import datetime
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, select_autoescape

from .annotations import describe
from .context import context_change
from .demo_meter import message_units, units
from .pricing import CATEGORIES, Prices, breakdown, summarize
from .schema import Run


def model_metrics(models, prices: Prices) -> dict:
    """Shared category totals for graph subtrees, user turns, and model calls."""
    metrics = {}
    parts = [breakdown(m, prices) for m in models]
    metrics["cells"] = [
        {
            "tokens": sum(p[i]["tokens"] or 0 for p in parts),
            "usd": sum((p[i]["usd"] or Decimal(0) for p in parts), Decimal(0)),
            "partial": any(
                p[i]["usd"] is None or p[i]["tokens"] is None for p in parts
            ),
        }
        for i in range(len(CATEGORIES))
    ]
    price_dates = set()
    for model in models:
        key = f"{model.provider}:{model.model}"
        rate = prices.models.get(prices.aliases.get(key, key))
        if rate:
            price_dates.add(rate.as_of or prices.as_of)
    metrics["price_dates"] = sorted(price_dates)
    metrics["stale_prices"] = any(
        d < datetime.now().astimezone().date() for d in price_dates
    )
    metrics["has_usage"] = bool(models)
    metrics["total"] = sum((c["usd"] for c in metrics["cells"]), Decimal(0))
    metrics["partial"] = any(c["partial"] for c in metrics["cells"])
    return metrics


def tree_rows(run: Run, prices: Prices) -> list[dict]:
    children = {}
    for step in sorted(run.steps, key=lambda s: s.start_ns):
        children.setdefault(step.parent_id, []).append(step)
    rows = []

    def visit(step, depth, parent):
        index = len(rows)
        row = {
            "step": step,
            "description": step.context.get("description")
            or describe(step.kind, step.name, step.context, {}),
            "index": index,
            "parent": parent,
            "depth": depth,
            "has_children": bool(children.get(step.id)),
        }
        rows.append(row)
        models = [step] if step.kind == "model" else []
        for child in children.get(step.id, []):
            models.extend(visit(child, depth + 1, index))
        row.update(model_metrics(models, prices))
        return models

    for root in children.get(None, []):
        visit(root, 0, None)
    return rows


def tool_request_tokens(content):
    if isinstance(content, str):
        try:
            content = literal_eval(content)
        except (ValueError, SyntaxError):
            pass
    return len(units(content))


def conversation_turns(run: Run, prices: Prices) -> list[dict]:
    turns = {}
    previous_models = {}
    request_number = 0
    for step in sorted(run.steps, key=lambda s: s.start_ns):
        if step.kind not in {"model", "tool"}:
            continue
        number = step.context.get("report_turn", 1)
        turn = turns.setdefault(
            number,
            {
                "number": number,
                "request": None,
                "events": [],
                "models": [],
            },
        )
        if step.kind == "model" and turn["request"] is None:
            turn["request"] = next(
                (
                    m.get("content")
                    for m in reversed(step.request)
                    if m.get("role") in {"human", "user"}
                ),
                None,
            )
        if step.kind == "model" and "request_tokens" not in turn:
            request = next(
                (
                    m
                    for m in reversed(step.request)
                    if m.get("role") in {"human", "user"}
                ),
                None,
            )
            turn["request_tokens"] = (
                len(message_units(request)) if run.demo and request else None
            )
        models = [step] if step.kind == "model" else []
        turn["models"].extend(models)
        metrics = model_metrics(models, prices)
        event = {"step": step, "cost": metrics["total"], **metrics}
        thinking = []
        if step.context.get("thinking_text"):
            thinking.append(step.context["thinking_text"])
        response_messages = []
        for message in step.response:
            content = message.get("content")
            if isinstance(content, list):
                visible = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {
                        "thinking",
                        "reasoning",
                    }:
                        value = (
                            block.get("thinking")
                            or block.get("reasoning")
                            or block.get("text")
                        )
                        if isinstance(value, str):
                            thinking.append(value)
                    elif not (
                        isinstance(block, dict)
                        and block.get("type") == "redacted_thinking"
                    ):
                        visible.append(block)
                message = {**message, "content": visible}
            response_messages.append(message)
        event["thinking_text"] = "\n".join(thinking)
        event["response_messages"] = response_messages
        if step.kind == "model":
            request_number += 1
            event["request_label"] = f"R{request_number}"
            # Compare sequential calls on the same named graph path, across user turns.
            by_id = {s.id: s for s in run.steps}
            ancestry = []
            parent = step.parent_id
            while parent:
                ancestry.append(by_id[parent].name)
                parent = by_id[parent].parent_id
            key = (step.provider, step.model, tuple(ancestry))
            event["growth"] = context_change(step, previous_models.get(key))
            previous = previous_models.get(key)
            tail = step.request[event["growth"]["retained"] :]
            event["input_parts"] = []
            if previous:
                for role, label in [
                    ("ai", "Assistant / tool-call input"),
                    ("tool", "Tool-result input"),
                ]:
                    messages = [m for m in tail if m.get("role") == role]
                    if messages:
                        event["input_parts"].append(
                            {
                                "label": label,
                                "tokens": sum(len(message_units(m)) for m in messages)
                                if run.demo
                                else None,
                            }
                        )
            ledger = json.loads(step.context.get("context_ledger", "{}"))
            event["request_cache_write"] = ledger.get("request_cache_write_tokens")
            event["request_context_total"] = ledger.get("request_tokens")
            event["response_context_total"] = ledger.get(
                "context_after_response_tokens"
            )
            event["response_cache_write"] = ledger.get("response_cache_write_tokens")
            event["has_tool_calls"] = any(m.get("tool_calls") for m in step.response)
            previous_models[key] = step
        elif step.response:
            event["result_chars"] = sum(
                len(str(m.get("content", ""))) for m in step.response
            )
            event["result_units"] = (
                sum(len(message_units(m)) for m in step.response) if run.demo else None
            )
        event["request_tokens"] = [
            tool_request_tokens(m.get("content")) if run.demo else None
            for m in step.request
        ]
        event["response_tokens"] = [
            step.usage.output_tokens
            if step.kind == "model" and step.usage and len(step.response) == 1
            else len(message_units(m))
            if run.demo
            else None
            for m in step.response
        ]
        event["call_tokens"] = {
            call.get("id"): len(units(call.get("args"))) if run.demo else None
            for m in step.response
            for call in m.get("tool_calls", [])
        }
        turn["events"].append(event)
    for turn in turns.values():
        turn.update(model_metrics(turn.pop("models"), prices))
    return list(turns.values())


def cost_chart(turns, prices):
    """Project the same per-call accounting into a shared EUR axis."""
    if not prices.exchange:
        return None
    colors = [
        "#2458a6",
        "#35a6a0",
        "#ce9331",
        "#915fc0",
        "#dc7952",
        "#658d35",
        "#c25989",
    ]
    bars = []
    for turn in turns:
        for event in turn["events"]:
            if event["step"].kind != "model":
                continue
            bars.append(
                {
                    "label": f"Request {len(bars) + 1}",
                    "turn": turn["number"],
                    "id": event["step"].id,
                    "partial": event["partial"],
                    "segments": [
                        {
                            "label": CATEGORIES[i][1],
                            "tokens": cell["tokens"],
                            "eur": cell["usd"] * prices.exchange.rate,
                            "color": colors[i],
                        }
                        for i, cell in enumerate(event["cells"])
                    ],
                    "total": event["total"] * prices.exchange.rate,
                }
            )
    cumulative = Decimal(0)
    for bar in bars:
        cumulative += bar["total"]
        bar["cumulative"] = cumulative
    maximum = cumulative
    scale = maximum * Decimal("1.15") if maximum else Decimal(1)
    bar_width = 32
    cursor = 85
    groups = []
    for i, bar in enumerate(bars):
        if i and bar["turn"] != bars[i - 1]["turn"]:
            cursor += 56
        if not groups or groups[-1]["turn"] != bar["turn"]:
            groups.append({"turn": bar["turn"], "start": cursor})
        bar["x"] = cursor
        bar["center"] = cursor + bar_width / 2
        bar["short_label"] = f"R{i + 1}"
        groups[-1]["end"] = cursor + bar_width
        cursor += bar_width + 8
    width = max(680, cursor + 25)
    for bar in bars:
        accumulated = 0.0
        for segment in bar["segments"]:
            height = float(segment["eur"] / scale) * 260
            segment.update(height=height, y=290 - accumulated - height)
            accumulated += height
        bar["top"] = 290 - accumulated
        bar["cumulative_y"] = 290 - float(bar["cumulative"] / scale) * 260
    return {
        "bars": bars,
        "width": width,
        "bar_width": bar_width,
        "groups": groups,
        "line_points": " ".join(f"{b['center']},{b['cumulative_y']}" for b in bars),
        "ticks": [{"y": 290 - i * 65, "value": scale * i / 4} for i in range(5)],
        "legend": [
            {"label": label, "color": colors[i]}
            for i, (_, label) in enumerate(CATEGORIES)
            if any(b["segments"][i]["eur"] for b in bars)
        ],
        "partial": any(b["partial"] for b in bars),
    }


def render(run: Run, prices: Prices, destination: Path):
    if prices.currency != "USD":
        raise ValueError("EUR conversion currently requires a USD pricing table")
    env = Environment(autoescape=select_autoescape(default=True))
    template = env.from_string(
        files("lg_report").joinpath("templates/report.html").read_text()
    )
    turns = conversation_turns(run, prices)
    destination.write_text(
        template.render(
            run=run,
            prices=prices,
            summary=summarize(run, prices),
            rows=tree_rows(run, prices),
            turns=turns,
            chart=cost_chart(turns, prices),
            categories=[
                entry
                for entry in CATEGORIES
                if entry[0] not in {"cache_write_5m", "cache_write_1h"}
            ],
            all_categories=CATEGORIES,
            visible_token_indices=[
                i
                for i, (key, _) in enumerate(CATEGORIES)
                if key not in {"cache_write_5m", "cache_write_1h"}
            ],
            today=datetime.now().astimezone().date(),
        ),
        encoding="utf-8",
    )
