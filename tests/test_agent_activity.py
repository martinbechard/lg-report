"""Check agent ownership and HTML visibility using saved trace relationships.

No provider calls are required. Nested and overlapping invocations must retain
separate identities, and missing usage must remain an incomplete subtotal.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from decimal import Decimal
from pathlib import Path

from lg_report.report.collaboration import collaboration_diagrams
from lg_report.report.pricing import load_prices
from lg_report.report.render import agent_activity, conversation_turns, render
from lg_report.report.schema import Run, Step, Usage


def test_nested_agents_with_shared_model_and_omitted_content(tmp_path):
    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")

    def step(id, parent, name, kind="workflow", start=1, status="ok", usage=None):
        return Step(
            id=id,
            parent_id=parent,
            name=name,
            kind=kind,
            start_ns=start,
            end_ns=100,
            status=status,
            usage=usage,
            provider="demo" if kind == "model" else None,
            model="scripted-chat" if kind == "model" else None,
            context={"langgraph_node": name} if name in {"model", "tools"} else {},
        )

    run = Run(
        id="parent",
        title="Agents",
        status="ok",
        steps=[
            step("parent", None, "dispatcher"),
            step("node", "parent", "model"),
            step(
                "m1",
                "node",
                "shared-model",
                "model",
                usage=Usage(input_tokens=100, output_tokens=10),
            ),
            step("tools", "parent", "tools"),
            step("task1", "tools", "task", "tool", start=2),
            step("child1", "task1", "<specialist & one>", start=3),
            step("node2", "child1", "model", start=4),
            step("m2", "node2", "shared-model", "model", start=5),
            step("task2", "tools", "task", "tool", start=3),
            step("child2", "task2", "specialist two", start=4, status="error"),
            step("nested_task", "child1", "task", "tool", start=6),
            step(
                "nested",
                "nested_task",
                "nested specialist",
                start=7,
                status="incomplete",
            ),
        ],
    )
    data = agent_activity(run, prices)
    activities = {a["step"].id: a for a in data["activities"]}
    assert set(activities) == {"parent", "child1", "child2", "nested"}
    assert activities["parent"]["model_calls"] == 1
    assert activities["parent"]["total"] == Decimal("0.000056")
    assert activities["child1"]["partial"]
    assert activities["child1"]["caller"].id == "parent"
    assert activities["nested"]["caller"].id == "child1"
    assert not activities["child2"]["has_usage"]
    events = {
        e["step"].id: e for t in conversation_turns(run, prices) for e in t["events"]
    }
    assert events["m1"]["agent"].id == "parent"
    assert events["m2"]["agent"].id == "child1"
    assert [s.id for s in events["task1"]["delegated_agents"]] == ["child1"]
    output = tmp_path / "report.html"
    render(run, prices, output)
    html = output.read_text()
    assert "Agent activity" in html and "Delegated by" in html
    assert "&lt;specialist &amp; one&gt;" in html
    assert "<specialist & one>" not in html
    assert 'href="#agent-child1"' in html
    assert "Incomplete — known subtotal only" in html
    assert "Status: error" in html and "Status: incomplete" in html
    assert "No model calls recorded" in html
    assert (
        html.index('id="agent-activity-list"')
        < html.index('id="collaboration-heading"')
        < html.index("<h2>Conversation</h2>")
    )
    diagram = collaboration_diagrams(run, data, conversation_turns(run, prices))[0]
    assert len(diagram["lanes"]) == 4
    arrows = [e for e in diagram["events"] if e["source"] != e["target"]]
    assert [
        (e["source"], e["target"]) for e in arrows if e["label"] == "Delegate via task"
    ] == [("parent", "child1"), ("parent", "child2"), ("child1", "nested")]
    assert [
        (e["source"], e["target"]) for e in arrows if e["label"] == "Return from task"
    ] == [("child1", "parent")]
    assert [e["timestamp"] for e in diagram["events"]] == sorted(
        e["timestamp"] for e in diagram["events"]
    )
    assert any(
        "R2 · LLM" in e["label"] and e["source"] == "child1" for e in diagram["events"]
    )


def test_bare_model_has_no_invented_agent():
    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")
    run = Run(
        id="r",
        title="Bare model",
        status="ok",
        steps=[
            Step(id="m", name="model", kind="model", start_ns=1, end_ns=2, status="ok")
        ],
    )
    assert agent_activity(run, prices)["activities"] == []
    assert conversation_turns(run, prices)[0]["events"][0]["agent"] is None


def test_collapsible_sections_preserve_nested_agents(tmp_path):
    from html.parser import HTMLParser

    class DisclosureParser(HTMLParser):
        """Check native disclosure structure without requiring a browser runtime."""

        def __init__(self):
            super().__init__()
            self.stack = []
            self.details = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "details":
                self.details.append(
                    (attrs, [a for t, a in self.stack if t == "details"])
                )
                if attrs.get("class") == "tool-definitions":
                    assert "open" not in attrs
                else:
                    assert "open" in attrs or "class" not in attrs
            if tag == "summary":
                assert self.stack[-1][0] == "details"
            if tag not in {"meta", "br", "link", "input", "img", "hr"}:
                self.stack.append((tag, attrs))

        def handle_endtag(self, tag):
            assert self.stack[-1][0] == tag
            self.stack.pop()

    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")
    run = Run(
        id="r",
        title="Nested",
        status="ok",
        steps=[
            Step(
                id="p",
                name="parent",
                kind="workflow",
                start_ns=1,
                end_ns=10,
                status="ok",
            ),
            Step(
                id="m",
                parent_id="p",
                name="llm",
                kind="model",
                start_ns=2,
                end_ns=3,
                status="ok",
            ),
            Step(
                id="t",
                parent_id="p",
                name="task",
                kind="tool",
                start_ns=4,
                end_ns=9,
                status="ok",
            ),
            Step(
                id="c",
                parent_id="t",
                name="child",
                kind="workflow",
                start_ns=5,
                end_ns=8,
                status="ok",
            ),
            Step(
                id="cm",
                parent_id="c",
                name="llm",
                kind="model",
                start_ns=6,
                end_ns=7,
                status="ok",
            ),
        ],
    )
    activity = agent_activity(run, prices)
    assert [a["step"].id for a in activity["roots"]] == ["p"]
    assert activity["roots"][0]["children"][0]["step"].id == "c"
    output = tmp_path / "report.html"
    render(run, prices, output)
    parser = DisclosureParser()
    parser.feed(output.read_text())
    assert not parser.stack
    _, parents = next(d for d in parser.details if d[0].get("id") == "agent-c")
    assert any(a.get("id") == "agent-p" for a in parents)
    conversations = [
        d for d in parser.details if d[0].get("class") == "conversation-agent"
    ]
    assert len(conversations) == 2
    assert any("tool-event" in a.get("class", "") for a in conversations[1][1])
    assert all(
        any(a.get("class") == "turn" for a in parents) for _, parents in conversations
    )


def test_collaboration_keeps_repeated_and_overlapping_invocations_separate():
    steps = [
        Step(id="a", name="author", kind="workflow", start_ns=1, end_ns=8, status="ok"),
        Step(
            id="b", name="judge", kind="workflow", start_ns=2, end_ns=4, status="error"
        ),
        Step(
            id="c", name="author", kind="workflow", start_ns=9, end_ns=10, status="ok"
        ),
    ]
    run = Run(id="r", title="Repeated roles", status="ok", steps=steps)
    agents = {
        "activities": [
            {"step": step, "caller": None, "delegation": None, "model_calls": 0}
            for step in steps
        ]
    }
    diagram = collaboration_diagrams(run, agents, [])[0]
    assert [lane["step"].id for lane in diagram["lanes"]] == ["a", "b", "c"]
    # Temporal succession must not invent communication or approval.
    assert all(e["source"] == e["target"] for e in diagram["events"])
    assert not any("Approved" in e["label"] for e in diagram["events"])
    assert collaboration_diagrams(run, {"activities": []}, []) == []
