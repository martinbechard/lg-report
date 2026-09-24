"""Write editable Excel reports using the project's ordinary Python dependencies.

The excel_data projection owns trace interpretation and accounting. This module
lays out that evidence, links charge rows with formulas, and supplies calculated
cached values so readers without a calculation engine do not see fake zeros.
Excel recalculates after editing rates or the execution count. No desktop agent,
Node runtime, provider call, or network lookup is needed to export a workbook.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

UNKNOWN = "Unknown"
CATEGORIES = (
    ("Input at standard rate", (0,)),
    ("Cache read", (1,)),
    ("Cache write", (2, 3, 4)),
    ("Output (non-reasoning)", (5,)),
    ("Reasoning", (6,)),
)
MONEY_FORMAT = "0.###############;-0.###############;0"


def number(value):
    """Preserve absent evidence instead of converting it to a zero charge."""
    return UNKNOWN if value is None else Decimal(str(value))


def known_sum(values):
    """A total is complete only when every contributing value is known."""
    return UNKNOWN if UNKNOWN in values else sum(values, Decimal(0))


def message_text(messages):
    """Retain message content and tool arguments as literal, readable cell text."""
    parts = []
    for message in messages or []:
        content = message.get("content")
        if content:
            parts.append(content if isinstance(content, str) else json.dumps(content))
        parts.extend(json.dumps(call) for call in message.get("tool_calls", []))
    return "\n".join(parts)


class ReportSheet:
    """Pair Excel formulas with cached values from the same saved accounting data.

    One-based row numbers match Excel addresses. Column indexes are zero-based.
    Only formula() creates formulas; all captured strings remain literal text.
    The cache supports guarded cross-sheet totals without a private spreadsheet
    runtime. It is not a general formula evaluator.
    """

    def __init__(self, workbook, name, headers):
        self.sheet = workbook.add_worksheet(name)
        self.name = name
        self.values = {}
        base = {"font_name": "Arial", "font_size": 10, "valign": "top"}
        self.plain = workbook.add_format(base)
        self.money = workbook.add_format({**base, "num_format": MONEY_FORMAT})
        self.integer = workbook.add_format({**base, "num_format": "#,##0"})
        self.content = workbook.add_format({**base, "text_wrap": True})
        self.heading = workbook.add_format(
            {**base, "bold": True, "bg_color": "#DBE5F1", "font_color": "#173B70"}
        )
        header = workbook.add_format(
            {
                **base,
                "bold": True,
                "bg_color": "#173B70",
                "font_color": "white",
                "text_wrap": True,
            }
        )
        self.sheet.hide_gridlines(2)
        self.sheet.freeze_panes(1, 1)
        self.sheet.set_tab_color("#173B70")
        self.sheet.set_default_row(24)
        self.sheet.set_column(0, len(headers) - 1, 18, self.plain)
        self.sheet.set_row(0, 44)
        for column, label in enumerate(headers):
            self.value(1, column, label, header)

    def value(self, row, column, value, fmt=None):
        """Write data without interpreting formulas or URLs in captured content."""
        self.values[row, column] = value
        if isinstance(value, str):
            # Excel limits cell text to 32,767 characters. Make truncation visible;
            # the complete captured content remains in run.json and HTML.
            if len(value) > 32767:
                value = value[:32700] + "\n[Truncated for Excel; see run.json]"
            self.sheet.write_string(row - 1, column, value, fmt)
        elif value is not None:
            self.sheet.write(row - 1, column, value, fmt)

    def formula(self, row, column, expression, cached, fmt=None):
        """Store a calculated result as well as a formula for later Excel edits."""
        self.values[row, column] = cached
        self.sheet.write_formula(
            row - 1,
            column,
            expression,
            fmt or self.money,
            float(cached) if isinstance(cached, Decimal) else cached,
        )

    def total(self, row, column, sources):
        """Sum only designated charge cells; text Unknown must propagate."""
        if not sources:
            self.value(row, column, 0, self.money)
            return
        refs = ",".join(
            f"'{sheet.name}'!{xl_col_to_name(col)}{r}" for sheet, r, col in sources
        )
        cached = known_sum(
            [sheet.values.get((r, col), UNKNOWN) for sheet, r, col in sources]
        )
        self.formula(
            row,
            column,
            f'=IF(COUNT({refs})={len(sources)},SUM({refs}),"Unknown")',
            cached,
        )

    def projected(self, row, eur=4, usd=5, projected_eur=6, projected_usd=7):
        """Scale precise costs before rounding, using Excel's half-up convention."""
        for source, target in ((eur, projected_eur), (usd, projected_usd)):
            value = self.values.get((row, source), UNKNOWN)
            cached = (
                UNKNOWN
                if value == UNKNOWN
                else (Decimal(value) * 100000).quantize(
                    Decimal(1), rounding=ROUND_HALF_UP
                )
            )
            cell = f"{xl_col_to_name(source)}{row}"
            self.formula(
                row,
                target,
                f"=IF(ISNUMBER({cell}),ROUND({cell}*'Reference data'!$B$2,0),\"Unknown\")",
                cached,
                self.integer,
            )


def reference_sheet(workbook, data):
    """Expose editable tariffs and conversion alongside their saved provenance."""
    ref = ReportSheet(workbook, "Reference data", ["Parameter", "Value", "Unit"])
    prices = data["prices"]
    exchange = prices.get("exchange") or {}
    settings = [
        ("Number of executions", 100000, "executions"),
        ("USD → EUR", number(exchange.get("rate")), "EUR per USD"),
        ("Cache duration", "5m", ""),
        ("Run ID", data["run"]["id"], ""),
        (
            "Recording",
            "Offline simulation" if data["run"]["demo"] else "Provider trace",
            "",
        ),
        ("FX reference date", exchange.get("date"), ""),
        ("FX fetched at", exchange.get("fetched_at"), "UTC"),
        ("FX source", exchange.get("source", "Unavailable"), ""),
        ("Rounding", "Only projected costs: ROUND(cost × executions, 0)", ""),
        (
            "Tree totals",
            "Parent rows include descendants; do not sum all tree rows.",
            "",
        ),
        (
            "Cache writes",
            "Simulated context movement is not a provider-billed write.",
            "",
        ),
        ("Pricing scope", prices.get("note"), ""),
    ]
    for row, values in enumerate(settings, 2):
        for col, value in enumerate(values):
            ref.value(row, col, value, ref.content)
    ref.sheet.data_validation(
        "B2", {"validate": "integer", "criteria": ">=", "value": 0}
    )
    ref.value(
        2,
        1,
        100000,
        workbook.add_format({"bg_color": "#FFF1B8", "num_format": "#,##0"}),
    )
    ref.value(3, 1, number(exchange.get("rate")), ref.money)
    headers = [
        "Model",
        "Input USD / 1M tokens",
        "Cache read USD / 1M tokens",
        "Cache write USD / 1M tokens",
        "Output USD / 1M tokens",
        "Reasoning USD / 1M tokens",
        "Verified date",
        "Source",
        "Notes",
    ]
    for col, label in enumerate(headers):
        ref.value(15, col, label, ref.heading)
    rate_rows = {}
    for row, (key, rate) in enumerate(prices["models"].items(), 16):
        rate_rows[key] = row
        values = [
            key,
            number(rate.get("input")),
            number(rate.get("cache_read")),
            number(
                rate.get("cache_write")
                if rate.get("cache_write") is not None
                else rate.get("cache_write_5m")
            ),
            number(rate.get("output")),
            number(rate.get("output")),
            rate.get("as_of") or prices.get("as_of"),
            rate.get("source") or "; ".join(prices.get("sources", [])),
            "Cache write assumes 5m when unspecified",
        ]
        for col, value in enumerate(values):
            ref.value(row, col, value, ref.money if 1 <= col <= 5 else ref.content)
        ref.sheet.set_row(row - 1, 72)
    ref.sheet.set_column("A:A", 34)
    ref.sheet.set_column("B:B", 52)
    ref.sheet.set_column("C:I", 26)
    ref.sheet.set_column("H:H", 65)
    return ref, rate_rows


def sequence_rows(events):
    """Lay out request/context/response evidence without charging context twice."""
    rows = [{"stage": "Run total", "kind": "total", "event": None}]
    turns, calls, categories = {}, {}, {}
    for event in events:
        step = event["step"]

        def add(stage, event=event, **details):
            """Return the future Excel row so formulas can target this evidence."""
            rows.append({"stage": stage, "event": event, **details})
            return len(rows) + 1

        if event["turn"] not in turns:
            turns[event["turn"]] = add(f"Turn {event['turn']}", kind="turn")
        if step["kind"] == "tool":
            add(f"Tool · {step['name']}", kind="tool")
            add("    Arguments", text=message_text(step.get("request")))
            add("    Result", text=message_text(step.get("response")))
            continue
        usage = step.get("usage") or {}
        context = step.get("context") or {}
        ledger = json.loads(context.get("context_ledger", "{}"))
        growth = event.get("growth") or {}
        charges = categories[step["id"]] = {}
        add("LLM request", tokens=usage.get("input_tokens"), kind="request")
        if usage.get("cache_read") or growth.get("previous_tokens") is not None:
            charges[1] = add("Conversation history · cache read", cat=1)
        charges[0] = add("Input at standard rate", cat=0)
        if growth.get("previous_tokens") is None:
            add(
                "    Tool definitions",
                tokens=context.get("simulated_definitions_tokens"),
            )
            add(
                "    System prompt",
                tokens=context.get("simulated_system_tokens"),
                text=message_text(
                    [m for m in step.get("request", []) if m["role"] == "system"]
                )
                or "Empty system message (framing only).",
            )
        if event.get("user_prompt"):
            add(
                "    User prompt",
                tokens=event.get("user_tokens"),
                text=event["user_prompt"],
            )
        tools = [
            m
            for m in step.get("request", [])[growth.get("retained", 0) :]
            if m["role"] == "tool"
        ]
        if tools:
            add(
                "    Tool result",
                text=message_text(tools),
                tokens=next(
                    (
                        p["tokens"]
                        for p in event.get("input_parts", [])
                        if p["label"] == "Tool-result input"
                    ),
                    None,
                ),
            )
        billed_write = any(
            event["cells"][i]["tokens"] or event["cells"][i]["partial"]
            for i in (2, 3, 4)
        )
        row = add(
            "Cache write",
            tokens=ledger.get("request_cache_write_tokens"),
            cat=2 if billed_write else None,
        )
        if billed_write:
            charges[2] = row
        add(
            "Context · added / total",
            tokens=ledger.get("request_cache_write_tokens"),
            total=ledger.get("request_tokens"),
        )
        add("LLM response", tokens=usage.get("output_tokens"), kind="response")
        if (
            event["cells"][6]["tokens"]
            or event["cells"][6]["partial"]
            or event.get("thinking")
        ):
            charges[4] = add("Reasoning", cat=4)
            if event.get("thinking"):
                add("    Reasoning text", text=event["thinking"], kind="thinking")
        charges[3] = add("Output", cat=3)
        add("    Output content", text=message_text(step.get("response")))
        add("Cache write", tokens=ledger.get("response_cache_write_tokens"))
        add(
            "Context · added / total",
            tokens=ledger.get("response_cache_write_tokens"),
            total=ledger.get("context_after_response_tokens"),
        )
        calls[step["id"]] = add("Call total", kind="calltotal")
    return rows, turns, calls, categories


def write_charges(sheet, row, event, category, ref, rate_rows, data):
    """Price a single charge row; retain exact lifetime-specific cache charges."""
    step = event["step"]
    indices = CATEGORIES[category][1]
    cells = [event["cells"][i] for i in indices]
    tokens = sum(c["tokens"] for c in cells) if step.get("usage") else UNKNOWN
    sheet.value(row, 3, tokens, sheet.integer)
    key = f"{step['provider']}:{step['model']}"
    tariff = rate_rows.get(data["prices"].get("aliases", {}).get(key, key))
    usd = UNKNOWN
    if tariff and not any(c["partial"] for c in cells):
        usd = sum((Decimal(str(c["usd"])) for c in cells), Decimal(0))
        usage = step.get("usage") or {}
        if category == 2 and (
            usage.get("cache_write_1h") or usage.get("cache_write_5m")
        ):
            sheet.value(row, 5, usd, sheet.money)
        else:
            rate_cell = f"'Reference data'!{xl_col_to_name(category + 1)}{tariff}"
            sheet.formula(
                row,
                5,
                f'=IF(D{row}=0,0,IF(ISNUMBER({rate_cell}),D{row}*{rate_cell}/1000000,"Unknown"))',
                usd,
            )
    else:
        sheet.value(row, 5, UNKNOWN)
    fx = ref.values[3, 1]
    eur = UNKNOWN if UNKNOWN in (usd, fx) else usd * fx
    sheet.formula(
        row,
        4,
        f"=IF(AND(ISNUMBER(F{row}),ISNUMBER('Reference data'!$B$3)),F{row}*'Reference data'!$B$3,\"Unknown\")",
        eur,
    )
    sheet.projected(row)


def export_workbook(data, output):
    """Export the three report views with exact cached accounting and editable formulas.

    Missing charges propagate Unknown. Complete run totals must agree with the
    shared Python projection before the workbook is accepted. No model executes.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with xlsxwriter.Workbook(
        output, {"strings_to_formulas": False, "strings_to_urls": False}
    ) as workbook:
        turns = ReportSheet(
            workbook,
            "Turns",
            [
                "Request",
                "Turn",
                "Sequence",
                "Tokens",
                "EUR / execution",
                "USD / execution",
                "Projected EUR",
                "Projected USD",
                "Content",
                "Model · effort",
                "Status",
                "Context total tokens",
                "Span ID",
            ],
        )
        tree_headers = [
            "Operation",
            "Depth",
            "Type",
            "Model · effort",
            "Status",
            "Input context tokens",
            "Description",
            "Span ID",
            "Parent ID",
        ]
        tree_headers += [
            f"{label} {unit}"
            for label, _ in CATEGORIES
            for unit in ("tokens", "EUR", "USD")
        ]
        tree_headers += [
            "Total EUR",
            "Total USD",
            "Projected EUR",
            "Projected USD",
            "Request",
            "Elapsed ms",
            "Accounting",
        ]
        tree = ReportSheet(workbook, "Execution tree", tree_headers)
        ref, rate_rows = reference_sheet(workbook, data)
        rows, turn_rows, calls, categories = sequence_rows(data["events"])
        for row, detail in enumerate(rows, 2):
            event = detail["event"] or {}
            step = event.get("step", {})
            kind = detail.get("kind")
            model = step.get("model") or ""
            if step.get("effort"):
                model += "-" + step["effort"]
            values = [
                event.get("label")
                if step.get("kind") == "model" and kind != "turn"
                else None,
                event.get("turn"),
                detail["stage"],
                detail.get("tokens"),
                None,
                None,
                None,
                None,
                detail.get("text"),
                model if kind == "request" else None,
                step.get("status") if kind in ("calltotal", "tool") else None,
                detail.get("total"),
                step.get("id") if kind in ("request", "tool") else None,
            ]
            if kind in ("total", "turn", "request", "response", "calltotal", "tool"):
                turns.sheet.set_row(row - 1, 24, turns.heading)
            for col, value in enumerate(values):
                turns.value(row, col, value, turns.content if col == 8 else None)
            if detail.get("text"):
                lines = sum(
                    max(1, (len(line) + 87) // 88)
                    for line in str(detail["text"]).splitlines()
                )
                turns.sheet.set_row(row - 1, min(409, max(24, lines * 13 + 8)))
            if detail.get("cat") is not None:
                write_charges(turns, row, event, detail["cat"], ref, rate_rows, data)
            if kind == "calltotal":
                for col in (4, 5):
                    turns.total(
                        row,
                        col,
                        [(turns, r, col) for r in categories[step["id"]].values()],
                    )
                turns.projected(row)
        for col in (4, 5):
            turns.total(2, col, [(turns, r, col) for r in calls.values()])
        turns.projected(2)
        for turn, row in turn_rows.items():
            selected = [
                calls[e["step"]["id"]]
                for e in data["events"]
                if e["turn"] == turn and e["step"]["kind"] == "model"
            ]
            for col in (4, 5):
                turns.total(row, col, [(turns, r, col) for r in selected])
            turns.projected(row)
        actual = turns.values[2, 5]
        if actual != UNKNOWN and abs(
            actual - Decimal(str(data["expected_usd"]))
        ) > Decimal("1e-10"):
            raise ValueError(
                "Workbook total differs from the saved accounting projection"
            )
        by_id = {t["step"]["id"]: t["step"] for t in data["tree"]}
        for row, item in enumerate(data["tree"], 2):
            step = item["step"]
            ids = []
            for model_id in calls:
                current = model_id
                while current:
                    if current == step["id"]:
                        ids.append(model_id)
                        break
                    current = by_id.get(current, {}).get("parent_id")
            values = [
                "    " * item["depth"] + step["name"],
                item["depth"],
                step["kind"],
                (step.get("model") or "")
                + ("-" + step["effort"] if step.get("effort") else ""),
                step["status"],
                (step.get("usage") or {}).get("input_tokens"),
                item["description"],
                step["id"],
                step.get("parent_id"),
            ]
            for col, value in enumerate(values):
                tree.value(row, col, value, tree.content if col in (0, 6) else None)
            tree.sheet.set_row(row - 1, 54)
            if ids:
                for category in range(5):
                    sources = [
                        categories[i][category]
                        for i in ids
                        if category in categories[i]
                    ]
                    for offset in range(3):
                        tree.total(
                            row,
                            9 + 3 * category + offset,
                            [(turns, r, 3 + offset) for r in sources],
                        )
                for col in (4, 5):
                    tree.total(row, 20 + col, [(turns, calls[i], col) for i in ids])
                tree.projected(row, 24, 25, 26, 27)
            event = next(
                (e for e in data["events"] if e["step"]["id"] == step["id"]), None
            )
            tree.value(
                row, 28, event["label"] if event and step["kind"] == "model" else None
            )
            tree.value(row, 29, (step["end_ns"] - step["start_ns"]) / 1e6)
            tree.value(
                row,
                30,
                "Own call"
                if step["kind"] == "model"
                else "Subtree total"
                if ids
                else "No model charge",
            )
        turns.sheet.set_column("A:B", 10)
        turns.sheet.set_column("C:C", 38)
        turns.sheet.set_column("E:F", 22, turns.money)
        turns.sheet.set_column("I:I", 100)
        turns.sheet.set_column("J:J", 26)
        turns.sheet.set_column("M:M", 38)
        tree.sheet.set_column("A:A", 48)
        tree.sheet.set_column("G:G", 62)
        tree.sheet.set_column("H:I", 38)
        for sheet, status_col, last in (
            (turns, "K", len(rows) + 1),
            (tree, "E", len(data["tree"]) + 1),
        ):
            if last >= 2:
                sheet.sheet.conditional_format(
                    f"A2:AE{last}",
                    {
                        "type": "formula",
                        "criteria": f'=${status_col}2="error"',
                        "format": workbook.add_format(
                            {"bg_color": "#FFD3D3", "font_color": "#8B1010"}
                        ),
                    },
                )
    return output


def main():
    """Export saved JSON with normal project dependencies; errors remain visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_workbook(json.loads(args.input.read_text()), args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
