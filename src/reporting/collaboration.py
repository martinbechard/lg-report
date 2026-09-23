"""Lay out recorded agent activity as offline SVG sequence diagrams.

Each named agent within a caller path owns a lifeline. Repeated invocations
share that lifeline and retain separate activation periods.
Arrows require recorded caller relationships; chronological proximity alone is
not evidence of a handoff. This projection never changes run data or accounting.

AI attribution: Modified with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from textwrap import wrap


def collaboration_diagrams(run, agents, turns):
    """Help readers follow who called whom during each recorded conversation turn.

    Return diagram dictionaries consumed by the HTML template: lanes identify
    named participants and events describe observed work or handoffs. ``run`` is
    normalized trace evidence, ``agents`` supplies resolved activities/callers,
    and ``turns`` supplies the renderer's grouped model/tool events. These are
    display projections, not live agents or executable scheduling instructions.

    Successful return arrows require both a completed child and completed task.
    Failed, interrupted, or unfinished invocations retain their observed status.
    Vertical spacing expresses event order, not proportional elapsed time.
    """
    start = min((s.start_ns for s in run.steps), default=0)
    groups = {}
    for activity in agents["activities"]:
        number = activity["step"].context.get("report_turn", 1)
        groups.setdefault(number, []).append(activity)
    # Names are the stable identity exposed by these traces; span IDs identify
    # calls, not agent objects. Include caller ancestry so equally named roles
    # belonging to different parents do not collapse into one participant.
    activities_by_id = {a["step"].id: a for a in agents["activities"]}

    def participant_key(activity):
        """Resolve a role's caller path without using per-invocation span IDs."""
        caller = activity["caller"]
        parent = activities_by_id.get(caller.id) if caller else None
        prefix = participant_key(parent) if parent else ()
        return (*prefix, activity["step"].name)

    diagrams = []
    for number, activities in groups.items():
        lanes = []
        by_key = {}
        by_id = {}
        for activity in activities:
            key = participant_key(activity)
            if key not in by_key:
                lane = {
                    "step": activity["step"],
                    "x": 220 + len(lanes) * 280,
                    "name_lines": wrap(activity["step"].name, 28) or ["Unnamed agent"],
                    "invocations": 0,
                    "model_calls": 0,
                }
                by_key[key] = lane
                lanes.append(lane)
            lane = by_key[key]
            lane["invocations"] += 1
            lane["model_calls"] += activity["model_calls"]
            by_id[activity["step"].id] = lane
        activations = []
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
            """Represent an observed action or handoff at the correct place in the diagram.

            The caller supplies the observed ``timestamp`` in Unix nanoseconds,
            display ``label``, and recorded ``status``. Return the appended event
            so activation boundaries follow the same layout; rendering happens later in the HTML template.

            ``source`` and ``target`` must be IDs present in ``by_id``; the
            self-target form is intentional for local work such as model/tool
            activity. ``order`` is a stable tie-breaker for equal timestamps,
            so diagram ordering never invents precision that the trace lacks.
            The helper mutates only this diagram's temporary ``events`` list.
            """
            event = {
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
            events.append(event)
            return event

        for activity in activities:
            step = activity["step"]
            caller = activity["caller"]
            task = activity["delegation"]
            # Draw a cross-lane call only when this turn contains the recorded
            # caller. A missing caller lane becomes a local start marker, never
            # a guessed link to the preceding chronological agent.
            if caller and caller.id in by_id:
                opened = add(
                    step.start_ns,
                    caller.id,
                    step.id,
                    "Delegate via task" if task else "Call agent",
                    "ok",
                    0,
                )
            else:
                opened = add(step.start_ns, step.id, step.id, "Started", "ok", 0)
            if (
                caller
                and caller.id in by_id
                and step.status == "ok"
                and (task is None or task.status == "ok")
            ):
                closed = add(
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
                closed = add(
                    step.end_ns,
                    step.id,
                    step.id,
                    f"{label} · {step.status}",
                    step.status,
                    2,
                )
            # Keep each call independently inspectable even when its lifeline
            # is shared. Waiting for human input lies between activations.
            activations.append(
                {
                    "step": step,
                    "x": by_id[step.id]["x"],
                    "opened": opened,
                    "closed": closed,
                }
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
        # Each event dictionary carries its clock time and phase order: calls
        # precede local work, which precedes returns at an identical timestamp.
        # This sorting lambda only selects that key; it does not run an action.
        events.sort(key=lambda event: (event["timestamp"], event["order"]))
        y = header_height + 30
        for event in events:
            event["lines"] = wrap(event["label"], 29)
            event["y"] = y
            event["center"] = (event["x1"] + event["x2"]) / 2
            event["box_height"] = 16 * len(event["lines"]) + 12
            y += event["box_height"] + 30
        for activation in activations:
            activation["y"] = activation["opened"]["y"]
            activation["height"] = max(1, activation["closed"]["y"] - activation["y"])
        diagrams.append(
            {
                "turn": number,
                "lanes": lanes,
                "events": events,
                "activations": activations,
                "width": 100 + 280 * len(lanes),
                "height": y + 10,
                "header_height": header_height,
            }
        )
    return diagrams
