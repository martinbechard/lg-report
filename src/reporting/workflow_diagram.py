"""Draw saved workflow topology as nested, self-contained SVG state diagrams.

The trace owns the definition; rendering never imports current workflow code or
infers state transitions from adjacent model calls. Containment shows nested
workflow ownership, solid arrows show declared transitions, and dashed links
expand a call node into its child graph. Repeated child definitions share a box;
two-headed task links show optional delegation and return. All saved text is escaped as SVG data.
Simple single-call workflows omit redundant Start/End markers in this view.
Runtime middleware sits above the separate agent-behavior boundary;
the native compiled edges remain visible, including hooks before model calls.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import re
from html import escape
from textwrap import wrap


def workflow_diagrams(run):
    """Render each distinct recorded definition once, including untaken branches.

    Older traces have no topology and therefore get no speculative diagram.
    Repeated turns using the same definition share a single diagram.
    """
    definitions = dict.fromkeys(
        step.context["report_workflow_definition"]
        for step in run.steps
        if step.context.get("report_workflow_definition")
    )
    return [_draw(json.loads(value), index) for index, value in enumerate(definitions)]


def _draw(definition, index):
    """Lay out the workflow and its explicit children inside nested boundaries."""
    containers, nodes, edges, links, boundaries = [], [], [], [], []
    marker = f"state-arrow-{index}"
    palette = ["#edf5fd", "#edf7f0", "#f5effb"]

    def layout(graph, x, y, depth):
        """Reserve one state column per graph; recursively enclose child graphs."""
        labels = {"__start__": "Start", "__end__": "End"}
        calls = [name for name in graph["nodes"] if name not in labels]
        # A straight-through call needs only its node and enclosing boundary.
        # Keep terminals when branches or loops convey additional behavior.
        single_call = (
            len(calls) == 1
            and len(graph["edges"]) == 2
            and {
                (edge["source"], edge["target"], edge["label"])
                for edge in graph["edges"]
            }
            == {("__start__", calls[0], ""), (calls[0], "__end__", "")}
        )
        visible_nodes = calls if single_call else graph["nodes"]
        # A compiled agent includes runtime hooks. Separate their ownership
        # visually without deleting hooks or inventing bypass transitions.
        middleware = {
            name for name in visible_nodes
            if graph.get("annotations", {}).get(name, {}).get("kind") == "middleware"
            or name == "SummarizationMiddleware.before_model"
        }
        column_width = 440
        positions = {}
        if middleware:
            # Entry belongs to the invocation, outside both ownership boxes.
            # Stack hooks above behavior; tool results still loop back to hooks.
            cursor = y + 116
            if "__start__" in visible_nodes:
                positions["__start__"] = (x + 100, cursor)
                cursor += 112
            for title, subtitle, members in (
                ("Runtime middleware", "Cross-cutting invocation hooks",
                 [name for name in visible_nodes if name in middleware]),
                ("Agent behavior", "Role instructions, model and tools",
                 [name for name in visible_nodes if name not in middleware and name not in labels]),
            ):
                top = cursor
                cursor += 48
                for name in members:
                    positions[name] = (x + 100, cursor)
                    cursor += 112
                boundaries.append((x + 20, top, 400, cursor - top - 20, title, subtitle))
                cursor += 12
            if "__end__" in visible_nodes:
                positions["__end__"] = (x + 100, cursor)
                cursor += 112
            bottom = cursor
        else:
            positions = {
                name: (x + 100, y + 140 + row * 112)
                for row, name in enumerate(visible_nodes)
            }
            bottom = y + 140 + len(positions) * 112
        for name in visible_nodes:
            nodes.append(
                (
                    name,
                    labels.get(name, name.replace("_", " ").capitalize()),
                    ({"kind": "middleware", "comment": "Before each model call: compact history if needed"}
                     if name in middleware else graph.get("annotations", {}).get(name, {})),
                    *positions[name],
                    name in graph["children"]
                    or any(
                        item["source"] == name for item in graph.get("delegations", [])
                    ),
                )
            )
        width = column_width
        child_y = y + 96
        child_positions = {}
        expansions = [(call, child, "") for call, child in graph["children"].items()]
        expansions.extend(
            (item["source"], item["graph"], item["tool"])
            for item in graph.get("delegations", [])
        )
        for call, child, tool in expansions:
            # This view describes definitions, not invocation counts. Repeated
            # calls to the same child definition share one box within their
            # parent. Compare the whole snapshot, including context and nested
            # children, so a matching name alone never merges different flows.
            key = json.dumps(child, sort_keys=True)
            if key not in child_positions:
                child_positions[key] = child_y
                child_width, child_bottom = layout(child, x + column_width, child_y, depth + 1)
                width = max(width, column_width + child_width + 20)
                bottom = max(bottom, child_bottom + 24)
                child_y = child_bottom + 24
            sx, sy = positions[call]
            links.append((sx + 210, sy + 34, x + column_width, child_positions[key] + 22, tool))
        containers.append((x, y, width, bottom - y, graph, depth))
        # Merge parallel routes while retaining every decision label. Multiple
        # decisions can intentionally share one terminal state.
        grouped = {}
        for edge in [] if single_call else graph["edges"]:
            grouped.setdefault((edge["source"], edge["target"]), []).append(
                edge["label"]
            )
        forward = backward = 0
        for (source, target), decisions in grouped.items():
            sx, sy = positions[source]
            tx, ty = positions[target]
            # Native agent routers often label a transition with its destination
            # name. That repeats the box text and can obscure other arrows;
            # retain only labels that add a decision beyond the destination.
            label = " / ".join(
                value for value in decisions if value and value != target
            )
            if source == target:
                # A model retry can transition to the same state. Give the
                # arrow a real loop instead of drawing overlapping straight lines.
                lane = max(sx, tx) + 242 + forward * 22
                forward += 1
                path = f"M {sx + 210} {sy + 12} H {lane} V {sy + 34} H {sx + 210}"
                lx, ly = lane + 5, sy + 26
            elif ty > sy and sx == tx and not any(
                sy < py < ty for _, py in positions.values()
            ):
                path = f"M {sx + 105} {sy + 68} V {ty}"
                lx, ly = sx + 115, sy + 68
            elif ty > sy:
                lane = max(sx, tx) + 242 + forward * 22
                forward += 1
                path = f"M {sx + 210} {sy + 34} H {lane} V {ty + 34} H {tx + 210}"
                lx, ly = lane + 5, (sy + ty) / 2 + 34
            else:
                lane = min(sx, tx) - 12 - backward * 12
                backward += 1
                path = f"M {sx} {sy + 34} H {lane} V {ty + 34} H {tx}"
                lx, ly = lane + 5, (sy + ty) / 2 + 34
            edges.append((path, label, lx, ly, source, target))
        return width, bottom

    width, bottom = layout(definition, 12, 12, 0)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width + 24} {bottom + 12}" '
            f'style="display:block;width:100%;min-width:1000px;max-width:{width + 24}px" role="img" '
            'aria-label="Workflow state diagram with nested workflow boundaries">'
        ),
        "<title>Workflow states and nested graph calls</title>",
        (
            "<desc>Solid arrows are possible state transitions. Dashed links expand "
            "child workflow calls. Runtime middleware is separate from agent behavior "
            "and intercepts model invocations. These are declared routes, not a recorded sequence.</desc>"
        ),
        (
            f'<defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#455568"/></marker></defs>'
        ),
    ]

    def text(x, y, value, **attrs):
        """Escape labels and emit only renderer-owned SVG attributes."""
        attributes = " ".join(
            f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items()
        )
        parts.append(f'<text x="{x}" y="{y}" {attributes}>{escape(str(value))}</text>')

    # Draw enclosing boundaries first so children and arrows stay visible.
    for x, y, w, h, graph, depth in sorted(containers, key=lambda item: item[-1]):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{palette[depth % len(palette)]}" stroke="#8595a7"/>'
        )
        text(x + 18, y + 28, graph["title"], font_size="17", font_weight="600")
        for line, content in enumerate(wrap(graph["context"], 48)):
            text(x + 18, y + 52 + line * 17, content, font_size="12")
    for x, y, w, h, title, subtitle in boundaries:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            'fill="none" stroke="#936d16" stroke-dasharray="4 3"/>'
        )
        text(x + 10, y + 17, title, font_size="13", font_weight="600")
        text(x + 10, y + 31, subtitle, font_size="10")
    for path, label, lx, ly, source, target in edges:
        parts.append(
            f'<path d="{path}" fill="none" stroke="#455568" stroke-width="1.5" '
            f'data-source="{escape(source, quote=True)}" data-target="{escape(target, quote=True)}" '
            f'marker-end="url(#{marker})"/>'
        )
        if label:
            text(
                lx,
                ly,
                label,
                font_size="11",
                paint_order="stroke",
                stroke="white",
                stroke_width="4",
                fill="#263749",
            )
    for sx, sy, tx, ty, tool in links:
        # Two arrowheads distinguish a model-selected task and its return from
        # an ordinary link that simply expands a workflow call's definition.
        start_marker = f' marker-start="url(#{marker})"' if tool else ""
        parts.append(
            f'<path d="M {sx} {sy} H {tx - 12} V {ty} H {tx}" '
            f'fill="none" stroke="#6c5993" stroke-dasharray="5 4"{start_marker} marker-end="url(#{marker})"/>'
        )
        if tool:
            text(sx + 8, sy - 10, f"{tool}: call / return", font_size="10")
    for name, label, annotation, x, y, child in nodes:
        # Native hook identifiers stay in the saved topology and arrow data.
        # Present the limit and hook timing separately so framework class names
        # do not become unreadable words in the learner-facing boxes.
        limit_hook = re.fullmatch(
            r"(Model|Tool)CallLimitMiddleware(?:\[(.+)\])?\.(before|after)_model",
            name,
        )
        if limit_hook:
            scope, tool, timing = limit_hook.groups()
            label = f"{scope} call limit"
            comment = f"{timing.capitalize()} model call"
            if tool:
                comment = f"{tool} · {comment}"
            annotation = {**annotation, "comment": comment}
        terminal = name in {"__start__", "__end__"}
        kind = annotation.get("kind", "subgraph" if child else "process")
        if kind == "model":
            label = "LLM call" if name == "model" else f"{label} · LLM call"
        # Distinct silhouettes describe responsibility, not extra graph states.
        parts.append(f'<g data-node-kind="{escape(kind, quote=True)}">')
        if limit_hook:
            parts.append(f'<title>{escape(name)}</title>')
        if kind == "middleware":
            parts.append(f'<rect x="{x}" y="{y}" width="210" height="68" rx="3" fill="#fff3d6" stroke="#936d16" stroke-dasharray="4 3"/>')
            label = "Context compaction hook"
        elif kind == "tool":
            parts.append(f'<path d="M {x + 12} {y} H {x + 210} L {x + 198} {y + 68} H {x} Z" fill="#e5f6ef" stroke="#34785c"/>')
        else:
            parts.append(
                f'<rect x="{x}" y="{y}" width="210" height="68" '
                f'rx="{22 if terminal else 12 if kind == "model" else 2}" '
                f'fill="{"#e9efff" if kind == "model" else "white"}" '
                f'stroke="{"#6c5993" if child else "#455568"}"/>'
            )
            if kind == "subgraph":
                parts.append(f'<path d="M {x + 7} {y} V {y + 68} M {x + 203} {y} V {y + 68}" stroke="#6c5993"/>')
        comment = annotation.get("comment", "")
        if comment:
            parts.append(f'<title>{escape(comment)}</title>')
            # Keep purpose comments inside their owning role or runtime lane.
            for row, line in enumerate(wrap(comment, 36)[:2]):
                text(x + 105, y + 43 + row * 12, line, text_anchor="middle", font_size="10")
        parts.append('</g>')
        # Middleware names are native graph states too; wrap them rather than
        # letting long framework labels escape their state boxes.
        lines = wrap(label, 28)
        for row, line in enumerate(lines):
            text(
                x + 105,
                y + (38 if terminal else 22) - (len(lines) - 1) * 7 + row * 14,
                line,
                text_anchor="middle",
                font_size="12",
            )
    parts.append("</svg>")
    return "".join(parts)
