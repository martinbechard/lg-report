"""Project recorded calls into conversation, agent activity, tree, and EUR chart views.

Rendering consumes saved usage and tariffs; it must not rerun an agent or fetch
new prices. The same accounting projections also feed the Excel exporter.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from ast import literal_eval
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, select_autoescape

from agent_runtime.harness.demo_meter import message_units, units
from reporting.annotations import describe
from reporting.collaboration import collaboration_diagrams
from reporting.context import context_change, context_utilization
from reporting.pricing import CATEGORIES, Prices, breakdown, summarize
from reporting.schema import Run


def model_metrics(models, prices: Prices) -> dict:
    """Give a report group its token totals, known costs, and completeness flags.

    ``models`` contains recorded model Steps selected for that group; ``prices``
    supplies tariffs. Return a metrics dictionary shared by table and chart views.

    Callers must pass each model once within this collection. Unknown values
    contribute no known amount but set partial=True, which consumers must retain;
    the numeric subtotal alone is not a complete estimate. No I/O occurs here.
    """
    # Here "models" means recorded Step objects, not runnable LLM instances.
    # Each nested list keeps CATEGORIES order so HTML and Excel agree on columns.
    model_steps = models
    metrics = {}
    category_costs_by_call = [breakdown(step, prices) for step in model_steps]
    metrics["cells"] = [
        {
            "tokens": sum(
                call_costs[i]["tokens"] or 0 for call_costs in category_costs_by_call
            ),
            "usd": sum(
                (
                    call_costs[i]["usd"] or Decimal(0)
                    for call_costs in category_costs_by_call
                ),
                Decimal(0),
            ),
            "partial": any(
                call_costs[i]["usd"] is None or call_costs[i]["tokens"] is None
                for call_costs in category_costs_by_call
            ),
        }
        for i in range(len(CATEGORIES))
    ]
    price_dates = set()
    for step in model_steps:
        key = f"{step.provider}:{step.model}"
        rate = prices.models.get(prices.aliases.get(key, key))
        # Only resolved tariffs carry meaningful verification dates. Missing
        # tariffs remain partial costs rather than receiving a fabricated date.
        if rate:
            price_dates.add(rate.as_of or prices.as_of)
    metrics["price_dates"] = sorted(price_dates)
    # This historical projection key means "contains model calls", not "all
    # calls reported usage". partial is the separate missing-evidence indicator.
    metrics["has_usage"] = bool(model_steps)
    metrics["total"] = sum((c["usd"] for c in metrics["cells"]), Decimal(0))
    metrics["partial"] = any(c["partial"] for c in metrics["cells"])
    return metrics


def tree_rows(run: Run, prices: Prices) -> list[dict]:
    """Let readers inspect where work occurred and what each subtree cost.

    ``run`` is a validated recording and ``prices`` its tariff snapshot. Return
    parent-before-child display rows with descendant model metrics attached.

    Parent totals describe containment, not additional billing: summing every row
    would count the same model repeatedly. Row indices and parent indices support
    HTML expansion; depth also gives Excel its indentation. Uses saved data only.
    """
    children = {}
    for step in sorted(run.steps, key=lambda s: s.start_ns):
        children.setdefault(step.parent_id, []).append(step)
    rows = []

    def visit(step, depth, parent):
        """Keep a subtree visible in execution order and attribute its contained costs.

        ``step`` is the current recorded span; ``depth`` determines indentation.
        Append its display row before visiting children, then return the model
        spans collected below it so its caller can aggregate ancestor metrics.

        parent is the containing row's index, not the callback span identifier.
        Returning model steps lets ancestors reuse the same evidence without
        confusing nesting with additional token consumption.
        """
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
        # Only model spans own charges; enclosing spans receive descendant
        # model metrics without adding a second charge of their own.
        models = [step] if step.kind == "model" else []
        for child in children.get(step.id, []):
            models.extend(visit(child, depth + 1, index))
        row.update(model_metrics(models, prices))
        return models

    for root in children.get(None, []):
        visit(root, 0, None)
    return rows


def execution_tree_view(rows, scopes):
    """Add a simplified HTML hierarchy without changing recorded spans or totals.

    Keep graph steps, agents, provider calls, tools, and every abnormal outcome.
    Adapters and internal model/tool nodes remain available in the full trace.
    Successful workflow layers containing only one visible agent/model are
    redundant in this view; this describes the recorded call structure, not
    Python object lifetime or a singleton design pattern.
    Hidden wrappers are bypassed when finding a visible parent so collapsing a
    branch works consistently in either view. Excel continues to use tree_rows.
    """
    for row in rows:
        step = row["step"]
        node = step.context.get("langgraph_node")
        row["detail_only"] = (
            step.status == "ok"
            and step.kind == "workflow"
            and row["parent"] is not None
            and step.id not in scopes
            and (step.name != node or node in {"model", "tools"})
        )
    # Resolve visible children from the leaves upward. A workflow with its own
    # preparation, tool, or pause step has more than a lone agent/model child
    # and must remain visible. Never remove an agent's identity or an abnormal
    # outcome merely because only one call happened beneath it.
    children = {row["index"]: [] for row in rows}
    for row in rows:
        if row["parent"] is not None:
            children[row["parent"]].append(row["index"])
    visible_children = {}
    for row in reversed(rows):
        contained = []
        for index in children[row["index"]]:
            contained.extend(visible_children[index])
        step = row["step"]
        if (
            not row["detail_only"]
            and step.kind == "workflow"
            and step.status == "ok"
            and step.id not in scopes
            and len(contained) == 1
        ):
            child = rows[contained[0]]["step"]
            if child.id in scopes or child.kind == "model":
                row["detail_only"] = True
        visible_children[row["index"]] = (
            contained if row["detail_only"] else [row["index"]]
        )

    for row in rows:
        parent = row["parent"]
        while parent is not None and rows[parent]["detail_only"]:
            parent = rows[parent]["parent"]
        row["summary_parent"] = parent
        row["summary_depth"] = (
            0 if parent is None else rows[parent]["summary_depth"] + 1
        )
        row["summary_children"] = False
        if not row["detail_only"] and parent is not None:
            rows[parent]["summary_children"] = True
    return rows


def agent_activity(run: Run, prices: Prices) -> dict:
    """Show which agents performed recorded work and assign each call one owner.

    ``run`` supplies validated span ancestry; ``prices`` prices owned calls.
    Return flat activities, a caller tree, scope lookup, and per-span owners.

    Named enclosing runnables identify agents; graph nodes such as model/tools
    are implementation steps, not separate agents. This also works on older
    recordings without message content. Each model belongs to its nearest scope
    so agent subtotals never charge delegated calls to their caller again.
    """
    by_id = {step.id: step for step in run.steps}

    def ancestors(step):
        """Find the enclosing agent or delegation responsible for a recorded step.

        ``step`` is a span in the enclosing run. Iteration yields parents nearest
        first; each yield returns control to the consuming ownership search.

        ``Run.check_tree`` has already rejected missing and cyclic parents, so
        this walk can focus on ownership projection. It yields schema objects
        rather than copying them, allowing callers to preserve the exact span
        status and metadata shown elsewhere in the report.
        """
        parent = step.parent_id
        while parent:
            ancestor = by_id[parent]
            yield ancestor
            parent = ancestor.parent_id

    def is_scope(step):
        """Decide whether a recorded span should identify an agent in the report.

        ``step`` is a normalized span. Return a Boolean used while selecting
        the nearest owner; this predicate does not run an agent.

        Graph step wrappers are filtered because they describe framework
        plumbing around a runnable. A named workflow or delegated invocation
        remains visible even when it has no model child, preserving failed or
        interrupted work in the activity view.
        """
        return (
            step.kind == "workflow"
            and (
                step.parent_id is None
                or step.name != step.context.get("langgraph_node")
            )
            and not any(
                tag.startswith("graph:step:") for tag in step.context.get("tags", [])
            )
        )

    scopes = {}
    owners = {}
    for step in run.steps:
        if step.kind == "model":
            scope = next((s for s in ancestors(step) if is_scope(s)), None)
            if scope:
                scopes[scope.id] = scope
                owners[step.id] = scope.id
        # A delegated graph can fail before its first model call. Keep that
        # invocation visible even though it has no billable descendants.
        if step.kind == "workflow" and step.parent_id:
            parent = by_id[step.parent_id]
            if parent.kind == "tool" and parent.name == "task":
                scopes[step.id] = step
    for step in run.steps:
        if step.id not in owners:
            owner = next((s for s in ancestors(step) if s.id in scopes), None)
            if owner:
                owners[step.id] = owner.id

    start = min((s.start_ns for s in run.steps), default=0)
    activities = []
    for scope in sorted(scopes.values(), key=lambda s: s.start_ns):
        models = [
            s for s in run.steps if s.kind == "model" and owners.get(s.id) == scope.id
        ]
        caller = next((s for s in ancestors(scope) if s.id in scopes), None)
        delegation = next(
            (s for s in ancestors(scope) if s.kind == "tool" and s.name == "task"),
            None,
        )
        activities.append(
            {
                "step": scope,
                "caller": caller,
                "delegation": delegation,
                "start_ms": (scope.start_ns - start) / 1_000_000,
                "model_calls": len(models),
                **model_metrics(models, prices),
            }
        )
    # Preserve the flat accounting projection while exposing a display tree.
    activity_by_id = {a["step"].id: {**a, "children": []} for a in activities}
    roots = []
    for activity in activity_by_id.values():
        caller = activity["caller"]
        if caller:
            activity_by_id[caller.id]["children"].append(activity)
        else:
            roots.append(activity)
    return {
        "activities": activities,
        "roots": roots,
        "scopes": scopes,
        "owners": owners,
    }


def tool_request_tokens(content):
    """Explain argument size in demo reports where provider telemetry is absent.

    ``content`` is captured argument data; return a simulator unit count for
    display, not an additional monetary charge.

    Captured arguments may be a Python-literal string; literal_eval normalizes
    that representation without executing code. Unparseable strings are counted
    as text. This word/punctuation count is never provider token telemetry.
    """
    # Tool arguments may be serialized Python literals or already structured;
    # parse only strings so structured values keep their original shape.
    if isinstance(content, str):
        try:
            content = literal_eval(content)
        except (ValueError, SyntaxError):
            pass
    return len(units(content))


def conversation_turns(run: Run, prices: Prices) -> list[dict]:
    """Let HTML and Excel tell the same chronological request-and-response story.

    ``run`` supplies normalized spans and ``prices`` their tariff snapshot.
    Return turn dictionaries containing display events, context comparisons,
    nested conversation nodes, and model-cost subtotals.

    Events follow span start time; model request numbers stay unique across turns.
    Each turn's cost includes its model calls exactly once, including subagents.
    Context comparisons use named graph ancestry so a specialist's independent
    history is not compared with its parent's history. Per-message estimates are
    available only for demos; real billing always uses captured provider usage.
    """
    turns = {}
    # A model identifier alone cannot identify a conversation: parent and expert
    # agents may share an LLM but must retain independent history baselines.
    previous_call_by_agent_path = {}
    request_number = 0
    agents = agent_activity(run, prices)
    for step in sorted(run.steps, key=lambda s: s.start_ns):
        # Framework workflow spans belong in the tree, not as additional
        # conversation messages that would obscure actual model/tool exchanges.
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
        # The first model with a captured user message establishes this turn's
        # prompt. Reverse search finds the latest user role, excluding system,
        # assistant and tool history; later calls must not overwrite it.
        if step.kind == "model" and turn["request"] is None:
            turn["request"] = next(
                (
                    m.get("content")
                    for m in reversed(step.request)
                    if m.get("role") in {"human", "user"}
                ),
                None,
            )
        # Initialize the prompt count once on a model event. Only a demo with a
        # captured prompt can use the simulator counter; real calls leave this
        # per-message count unknown because usage is reported per request.
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
        # Only model spans own charges; enclosing spans receive descendant
        # model metrics without adding a second charge of their own.
        models = [step] if step.kind == "model" else []
        turn["models"].extend(models)
        metrics = model_metrics(models, prices)
        owner = agents["scopes"].get(agents["owners"].get(step.id))
        event = {"step": step, "cost": metrics["total"], "agent": owner, **metrics}
        event["delegated_agents"] = [
            activity["step"]
            for activity in agents["activities"]
            if activity["delegation"] and activity["delegation"].id == step.id
        ]
        # Separate provider-exposed reasoning from visible output so templates can
        # truncate it without duplicating it in the response. Redacted blocks do
        # not contain displayable reasoning; never invent text from token counts.
        thinking = []
        # Include explicitly captured reasoning text only when present; a
        # reasoning-token count alone cannot reconstruct its content.
        if step.context.get("thinking_text"):
            thinking.append(step.context["thinking_text"])
        response_messages = []
        for message in step.response:
            content = message.get("content")
            # Block responses can mix visible text and reasoning; plain text is
            # already displayable and must not be interpreted as block metadata.
            if isinstance(content, list):
                visible = []
                for block in content:
                    # Only typed reasoning dictionaries are separated from the
                    # visible answer; arbitrary blocks remain response content.
                    if isinstance(block, dict) and block.get("type") in {
                        "thinking",
                        "reasoning",
                    }:
                        value = (
                            block.get("thinking")
                            or block.get("reasoning")
                            or block.get("text")
                        )
                        # Provider payloads may omit or redact the text. Display
                        # strings only, never stringify opaque reasoning objects.
                        if isinstance(value, str):
                            thinking.append(value)
                    # Redacted thinking is intentionally unavailable and should
                    # appear in neither the reasoning preview nor visible answer.
                    elif not (
                        isinstance(block, dict)
                        and block.get("type") == "redacted_thinking"
                    ):
                        visible.append(block)
                message = {**message, "content": visible}
            response_messages.append(message)
        event["thinking_text"] = "\n".join(thinking)
        event["response_messages"] = response_messages
        # Request numbering and context growth belong to LLM invocations; a tool
        # event below records its result without pretending to be another call.
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
            # Name paths remain stable between turns while callback IDs change.
            # This groups repeated invocations of the same named agent path;
            # it is not a provider cache key or proof of a cache hit.
            key = (step.provider, step.model, tuple(ancestry))
            event["growth"] = context_change(step, previous_call_by_agent_path.get(key))
            previous = previous_call_by_agent_path.get(key)
            tail = step.request[event["growth"]["retained"] :]
            event["input_parts"] = []
            # Only a prior call on this graph path establishes a retained prefix;
            # without it, setup is handled as first-request context. Match roles
            # in the new tail so old tool messages are not described as additions.
            if previous:
                for role, label in [
                    ("ai", "Assistant / tool-call input"),
                    ("tool", "Tool-result input"),
                ]:
                    messages = [m for m in tail if m.get("role") == role]
                    # Omit absent role groups rather than implying an empty
                    # assistant/tool addition. Counts are simulator-only; real
                    # telemetry does not supply these per-message estimates.
                    if messages:
                        event["input_parts"].append(
                            {
                                "label": label,
                                "tokens": sum(len(message_units(m)) for m in messages)
                                if run.demo
                                else None,
                            }
                        )
            # The simulator's ledger explains context movement, not extra billed
            # cache writes. Monetary totals still come solely from usage buckets.
            ledger = json.loads(step.context.get("context_ledger", "{}"))
            event["request_cache_write"] = ledger.get("request_cache_write_tokens")
            event["request_context_total"] = ledger.get("request_tokens")
            event["response_context_total"] = ledger.get(
                "context_after_response_tokens"
            )
            event["response_cache_write"] = ledger.get("response_cache_write_tokens")
            event["has_tool_calls"] = any(m.get("tool_calls") for m in step.response)
            previous_call_by_agent_path[key] = step
        # A tool with captured output can expose its result size; omitted output
        # has no measurable content. Only demos assign estimated token units.
        elif step.response:
            event["result_chars"] = sum(
                len(str(m.get("content", ""))) for m in step.response
            )
            event["result_units"] = (
                sum(len(message_units(m)) for m in step.response) if run.demo else None
            )
        # Auxiliary message/argument estimates are demo-only. A single model
        # response can use the provider's authoritative output total; multiple
        # responses cannot each claim that total, so only demos estimate them.
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
    by_id = {step.id: step for step in run.steps}
    for turn in turns.values():
        turn.update(model_metrics(turn.pop("models"), prices))
        # Keep the flat event list for accounting and charts. HTML additionally
        # nests tools and agent invocations by ancestry, skipping framework nodes.
        nodes = {}
        for index, event in enumerate(turn["events"]):
            event["first_in_turn"] = index == 0
            step = event["step"]
            nodes.setdefault(step.id, {"step": step, "children": []})["event"] = event
            parent = step.parent_id
            while parent:
                ancestor = by_id[parent]
                if parent in agents["scopes"] or ancestor.kind == "tool":
                    nodes.setdefault(parent, {"step": ancestor, "children": []})
                parent = ancestor.parent_id
        turn["conversation_nodes"] = []
        for node in sorted(nodes.values(), key=lambda n: n["step"].start_ns):
            parent = node["step"].parent_id
            while parent and parent not in nodes:
                parent = by_id[parent].parent_id
            if parent:
                nodes[parent]["children"].append(node)
            else:
                turn["conversation_nodes"].append(node)
    return list(turns.values())


def cost_chart(turns, prices):
    """Help readers see which requests drive cost and how costs accumulate.

    ``turns`` is the conversation projection; ``prices`` provides recorded FX.
    Return SVG geometry and labels for the template, or None without conversion.

    Each bar includes one model request and its response, split by billed token
    category; tools have no independent bar. Bars and the cumulative line share
    the EUR scale; context occupancy uses a separate percentage axis. Return None
    when FX is unavailable rather than labeling USD as EUR.
    Partial accounting remains flagged in the projection; no amounts are rounded.
    """
    # EUR geometry cannot be computed without a recorded conversion rate;
    # returning None lets the report state unavailable instead of plotting USD.
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
            # Tools have no independent token bill; their request/result tokens
            # are already in the model bars that emit and consume them.
            if event["step"].kind != "model":
                continue
            bars.append(
                {
                    "label": f"Request {len(bars) + 1}",
                    "turn": turn["number"],
                    "agent": event.get("agent"),
                    "id": event["step"].id,
                    "partial": event["partial"],
                    "context": context_utilization(event["step"], prices),
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
    # Nonzero totals reserve headroom above the cumulative line; an all-zero
    # chart uses a unit denominator to avoid division by zero in SVG geometry.
    scale = maximum * Decimal("1.15") if maximum else Decimal(1)
    # Tight spacing links requests within a turn; the larger gap marks the next
    # user turn without changing bar width or implying a different time scale.
    bar_width = 16
    cursor = 85
    groups = []
    for i, bar in enumerate(bars):
        # Only a real transition after the first bar merits the wider gap.
        if i and bar["turn"] != bars[i - 1]["turn"]:
            cursor += 28
        # Start a label group for the first bar or a new turn; otherwise extend
        # the existing group so its underline spans all calls in that turn.
        if not groups or groups[-1]["turn"] != bar["turn"]:
            groups.append({"turn": bar["turn"], "start": cursor})
        bar["x"] = cursor
        bar["center"] = cursor + bar_width / 2
        bar["short_label"] = f"R{i + 1}"
        groups[-1]["end"] = cursor + bar_width
        cursor += bar_width + 8
    # Vertical names sit directly under their request bars. Reserve enough
    # space for the longest name before drawing the shared turn bracket.
    for bar in bars:
        bar["agent_name"] = bar["agent"].name if bar["agent"] else "Unreported"
    turn_line_y = 334 + max((len(b["agent_name"]) for b in bars), default=0) * 6
    chart_height = turn_line_y + 40
    width = max(680, cursor + 85)
    # Keep the percent axis at least 0–100%; unusually large recorded inputs
    # expand it rather than clipping or silently capping the measured ratio.
    context_max = max(
        100, max((b["context"]["percent"] for b in bars if b["context"]), default=0)
    )
    context_lines = []
    context_points = []
    for bar in bars:
        accumulated = 0.0
        for segment in bar["segments"]:
            height = float(segment["eur"] / scale) * 260
            segment.update(height=height, y=290 - accumulated - height)
            accumulated += height
        bar["top"] = 290 - accumulated
        bar["cumulative_y"] = 290 - float(bar["cumulative"] / scale) * 260
        if bar["context"] is not None:
            bar["context_y"] = 290 - bar["context"]["percent"] / context_max * 260
            context_points.append(f"{bar['center']},{bar['context_y']}")
        elif context_points:
            # Unknown usage/capacity creates a real gap, not a zero or a line
            # interpolated through a request with no occupancy evidence.
            context_lines.append(" ".join(context_points))
            context_points = []
    if context_points:
        context_lines.append(" ".join(context_points))
    # Legend entries require a nonzero plotted segment somewhere; absent
    # categories would otherwise suggest bars contain costs they do not have.
    return {
        "bars": bars,
        "width": width,
        "bar_width": bar_width,
        "height": chart_height,
        "turn_line_y": turn_line_y,
        "groups": groups,
        "line_points": " ".join(f"{b['center']},{b['cumulative_y']}" for b in bars),
        "context_lines": context_lines,
        "context_ticks": [
            {"y": 290 - i * 65, "value": context_max * i / 4} for i in range(5)
        ],
        "context_missing": any(b["context"] is None for b in bars),
        "context_illustrative": any(
            b["context"] and b["context"]["illustrative"] for b in bars
        ),
        "ticks": [{"y": 290 - i * 65, "value": scale * i / 4} for i in range(5)],
        "legend": [
            {"label": label, "color": colors[i]}
            for i, (_, label) in enumerate(CATEGORIES)
            if any(b["segments"][i]["eur"] for b in bars)
        ],
        "partial": any(b["partial"] for b in bars),
    }


def render(run: Run, prices: Prices, destination: Path):
    """Make a recorded run inspectable in a browser with its accounting evidence.

    ``run`` is the normalized recording, ``prices`` supplies saved tariffs/FX,
    and ``destination`` receives the completed HTML file. Return no value.

    destination is overwritten and its parent must already exist. Reject a
    non-USD price table because conversion assumes USD→EUR. Autoescaping keeps
    captured prompts and tool output as data, not executable report markup.
    Template/read/write failures propagate to the caller.
    """
    # Every tariff conversion multiplies USD by the saved USD/EUR rate; reject
    # any other base currency instead of producing mislabeled amounts.
    if prices.currency != "USD":
        raise ValueError("EUR conversion currently requires a USD pricing table")
    env = Environment(autoescape=select_autoescape(default=True))
    template = env.from_string(
        files("reporting").joinpath("templates/report.html").read_text()
    )
    turns = conversation_turns(run, prices)
    agents = agent_activity(run, prices)
    # Lifetime-specific write buckets remain in accounting, but are excluded
    # from the requested compact table columns; the full category list still
    # drives charts and category-cost lookups.
    destination.write_text(
        template.render(
            run=run,
            prices=prices,
            summary=summarize(run, prices),
            rows=execution_tree_view(tree_rows(run, prices), agents["scopes"]),
            turns=turns,
            agents=agents,
            collaboration=collaboration_diagrams(run, agents, turns),
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
        ),
        encoding="utf-8",
    )
