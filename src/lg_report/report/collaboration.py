"""Lay out recorded agent activity as offline SVG sequence diagrams.

Each invocation owns a lane, including repeated agents and concurrent children.
Arrows require recorded caller relationships; chronological proximity alone is
not evidence of a handoff. This projection never changes run data or accounting.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from textwrap import wrap


def collaboration_diagrams(run, agents, turns):
    """Return per-turn lanes and timestamp-ordered activity for the HTML template.

    Successful return arrows require both a completed child and completed task.
    Failed, interrupted, or unfinished invocations retain their observed status.
    Vertical spacing expresses event order, not proportional elapsed time.
    """
    start = min((s.start_ns for s in run.steps), default=0)
    groups = {}
    for activity in agents["activities"]:
        number = activity["step"].context.get("report_turn", 1)
        groups.setdefault(number, []).append(activity)
    diagrams = []
    for number, activities in groups.items():
        lanes = []
        for index, activity in enumerate(activities):
            lanes.append(
                {
                    **activity,
                    "x": 220 + index * 280,
                    "name_lines": wrap(activity["step"].name, 28) or ["Unnamed agent"],
                }
            )
        by_id = {lane["step"].id: lane for lane in lanes}
        header_height = 65 + max(len(lane["name_lines"]) for lane in lanes) * 16
        events = []

        def add(
            timestamp,
            source,
            target,
            label,
            status,
            order,
            *,
            events=events,
            by_id=by_id,
        ):
            events.append(
                {
                    "timestamp": timestamp,
                    "source": source,
                    "target": target,
                    "label": label,
                    "status": status,
                    "order": order,
                    "time_ms": (timestamp - start) / 1_000_000,
                    "x1": by_id[source]["x"],
                    "x2": by_id[target]["x"],
                }
            )

        for activity in activities:
            step = activity["step"]
            caller = activity["caller"]
            task = activity["delegation"]
            if caller and caller.id in by_id:
                add(
                    step.start_ns,
                    caller.id,
                    step.id,
                    "Delegate via task" if task else "Call agent",
                    "ok",
                    0,
                )
            else:
                add(step.start_ns, step.id, step.id, "Started", "ok", 0)
            if (
                caller
                and caller.id in by_id
                and step.status == "ok"
                and (task is None or task.status == "ok")
            ):
                add(
                    task.end_ns if task else step.end_ns,
                    step.id,
                    caller.id,
                    "Return from task" if task else "Return to caller",
                    "ok",
                    2,
                )
            else:
                # Closing an incomplete span is recorder cleanup, not completion.
                label = "Finished" if step.status == "ok" else "Stopped / last recorded"
                add(
                    step.end_ns,
                    step.id,
                    step.id,
                    f"{label} · {step.status}",
                    step.status,
                    2,
                )
        for turn in turns:
            for event in turn["events"]:
                owner = event.get("agent")
                if turn["number"] != number or owner is None or owner.id not in by_id:
                    continue
                if event.get("delegated_agents"):
                    continue  # Already represented by delegation and return arrows.
                step = event["step"]
                label = (
                    f"{event['request_label']} · LLM"
                    if step.kind == "model"
                    else f"Tool · {step.name}"
                )
                add(
                    step.start_ns,
                    owner.id,
                    owner.id,
                    f"{label} · {step.status}",
                    step.status,
                    1,
                )
        events.sort(key=lambda event: (event["timestamp"], event["order"]))
        y = header_height + 30
        for event in events:
            event["lines"] = wrap(event["label"], 29)
            event["y"] = y
            event["center"] = (event["x1"] + event["x2"]) / 2
            event["box_height"] = 16 * len(event["lines"]) + 12
            y += event["box_height"] + 30
        diagrams.append(
            {
                "turn": number,
                "lanes": lanes,
                "events": events,
                "width": 100 + 280 * len(lanes),
                "height": y + 10,
                "header_height": header_height,
            }
        )
    return diagrams
