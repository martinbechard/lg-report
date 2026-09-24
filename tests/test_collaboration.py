"""Keep collaboration diagrams concise without concealing incomplete work.

Synthetic saved spans isolate the rendering rule from model behavior. Single
successful calls omit lifecycle boxes; additional work and errors retain them.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import pytest

from reporting.collaboration import collaboration_diagrams
from reporting.pricing import Prices
from reporting.render import agent_activity, conversation_turns
from reporting.schema import Run, Step


@pytest.mark.parametrize(
    "calls,status,tool,compact",
    [
        (1, "ok", False, True),
        (2, "ok", False, False),
        (1, "error", False, False),
        (1, "ok", True, False),
    ],
)
def test_single_call_lifecycle_boxes(calls, status, tool, compact):
    """Preserve failure markers and tool activity while simplifying one model call."""
    steps = [
        Step(
            id="agent",
            name="reviewer",
            kind="workflow",
            start_ns=0,
            end_ns=100,
            status=status,
        )
    ]
    steps.extend(
        Step(
            id=f"model-{i}",
            parent_id="agent",
            name="model",
            kind="model",
            start_ns=10 + i * 20,
            end_ns=20 + i * 20,
            status=status,
        )
        for i in range(calls)
    )
    if tool:
        steps.append(
            Step(
                id="tool",
                parent_id="agent",
                name="lookup",
                kind="tool",
                start_ns=70,
                end_ns=80,
                status="ok",
            )
        )
    run = Run(id="sample", title="One role", status=status, steps=steps)
    prices = Prices(
        currency="USD",
        models={},
        as_of="2026-09-23",
        note="No prices needed for diagram layout",
    )
    agents = agent_activity(run, prices)
    diagram = collaboration_diagrams(run, agents, conversation_turns(run, prices))[0]
    labels = [event["label"] for event in diagram["events"]]
    assert ("Started" not in labels) == compact
    assert len(diagram["activations"]) == 1
    assert diagram["activations"][0]["height"] > 0
    if status == "error":
        assert "Stopped / last recorded · error" in labels
    if tool:
        assert "Tool · lookup · ok" in labels


@pytest.mark.parametrize("edge,overlap,status,same_workflow,expected", [
    (True, False, "ok", True, 1),
    (False, False, "ok", True, 0),
    (True, True, "ok", True, 0),
    (True, False, "error", True, 0),
    (True, False, "ok", False, 0),
])
def test_workflow_progression_requires_recorded_edge(edge, overlap, status, same_workflow, expected):
    """Connect sequential workflow peers, never unrelated or overlapping roles."""
    import json

    topology = json.dumps({"nodes": ["plan", "work"], "edges": [
        {"source": "plan", "target": "work", "label": ""}
    ] if edge else []})
    steps = [Step(id="flow", name="flow", kind="workflow", start_ns=0, end_ns=100,
                  status="ok", context={"report_workflow_definition": topology})]
    if not same_workflow:
        steps.append(Step(id="other", name="flow", kind="workflow", start_ns=0, end_ns=100,
                          status="ok", context={"report_workflow_definition": topology}))
    for name, node, begin, end, parent, state in [
        ("planner", "plan", 1, 40, "flow", status),
        ("worker", "work", 30 if overlap else 50, 90, "flow" if same_workflow else "other", "ok"),
    ]:
        steps.extend([
            Step(id=name, parent_id=parent, name=name, kind="workflow", start_ns=begin,
                 end_ns=end, status=state, context={"langgraph_node": node}),
            Step(id=name + "-model", parent_id=name, name="model", kind="model",
                 start_ns=begin + 1, end_ns=end - 1, status=state),
        ])
    run = Run(id="progression", title="Workflow peers", status="ok", steps=steps)
    prices = Prices(models={}, as_of="2026-09-23", note="No billing needed")
    diagram = collaboration_diagrams(run, agent_activity(run, prices), conversation_turns(run, prices))[0]
    arrows = [event for event in diagram["events"] if event["source"] != event["target"]]
    assert len(arrows) == expected
    if arrows:
        assert (arrows[0]["source"], arrows[0]["target"], arrows[0]["label"], arrows[0]["order"]) == (
            "planner", "worker", "", 0,
        )
