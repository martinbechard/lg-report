"""Check agent ownership and HTML visibility using saved trace relationships.

No provider calls are required. Invocations retain separate evidence while
repeated agent roles share lifelines; missing usage remains an incomplete subtotal.

AI attribution: Modified with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from decimal import Decimal
from pathlib import Path

from reporting.collaboration import collaboration_diagrams
from reporting.pricing import load_prices
from reporting.render import agent_activity, conversation_turns, render
from reporting.schema import Run, Step, Usage


def test_quote_request_reports_native_agent_name(tmp_path):
    """Trace the real agent with a scripted decision to verify report ownership.

    The adapter name alone is insufficient: the nested native graph is the
    nearest owner of model calls, so its name must identify the quote role too.
    """
    from agent_runtime.workflows.quote_request import build_workflow
    from reporting.execute_runnable import execute_runnable
    from samples.quote_request.sample import DECISIONS, make_simulated_model

    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")
    directory = tmp_path / "quote"
    execute_runnable(
        build_workflow(make_simulated_model([DECISIONS[-1]])),
        {"values": {"description": "Print 500 invitations and addressed envelopes."}},
        directory,
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
    )
    run = Run.model_validate_json((directory / "run.json").read_text())
    activities = agent_activity(run, prices)["activities"]
    assert len(activities) == 1
    assert activities[0]["step"].name == "quote_interpreter"
    assert activities[0]["model_calls"] == 1
    events = conversation_turns(run, prices)[0]["events"]
    assert events[0]["agent"].name == "quote_interpreter"
    html = (directory / "report.html").read_text()
    assert f'id="agent-{activities[0]["step"].id}"' in html


# Use hand-built spans to protect ownership, partial-cost, escaping,
# and collaboration-diagram behavior without requiring a provider or network.
def test_nested_agents_with_shared_model_and_omitted_content(tmp_path):
    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")

    def step(id, parent, name, kind="workflow", start=1, status="ok", usage=None):
        # Make trace ownership testable without running agents.
        # `id` and `parent` define ancestry; `start` controls ordering. A missing
        # usage value deliberately models an unmetered call, not a free call.
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


# A standalone model span must remain visible as an event while not
# acquiring an imaginary agent identity from incomplete trace metadata.
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


# Parse the emitted HTML directly so disclosure nesting and default
# open/closed states remain covered even when no browser runtime is available.
def test_collapsible_sections_preserve_nested_agents(tmp_path):
    from html.parser import HTMLParser

    class DisclosureParser(HTMLParser):
        """Check native disclosure structure without requiring a browser runtime."""

        def __init__(self):
            # Start a fresh HTML nesting observation for this report.
            # The stack tracks open non-void elements; details stores each disclosure
            # and the ancestor disclosures present when it opened.
            super().__init__()
            self.stack = []
            self.details = []

        def handle_starttag(self, tag, attrs):
            # Catch malformed disclosure placement as the HTML parser visits a tag.
            # `attrs` is the parser-provided name/value sequence, not a DOM node;
            # record its ancestors before pushing this element onto the open stack.
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
            # Require closing tags to balance the emitted nesting.
            # The parser supplies the tag name; an unexpected close fails immediately
            # instead of allowing later ancestry checks to use a corrupted stack.
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


# Repeated role names and temporal overlap must not create invented
# approvals or communication edges in the collaboration view.
def test_collaboration_groups_repeated_roles_with_separate_activations():
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
    assert [lane["step"].id for lane in diagram["lanes"]] == ["a", "b"]
    assert [lane["invocations"] for lane in diagram["lanes"]] == [2, 1]
    activations = diagram["activations"]
    assert [a["step"].id for a in activations] == ["a", "b", "c"]
    assert activations[0]["x"] == activations[2]["x"]
    assert activations[0]["y"] + activations[0]["height"] < activations[2]["y"]
    assert activations[1]["step"].status == "error"
    # Temporal succession must not invent communication or approval.
    assert all(e["source"] == e["target"] for e in diagram["events"])
    assert not any("Approved" in e["label"] for e in diagram["events"])
    assert collaboration_diagrams(run, {"activities": []}, []) == []


def test_same_named_agents_under_different_callers_keep_distinct_lifelines():
    """A shared role name does not merge participants owned by distinct callers."""
    steps = [
        Step(id=id, name=name, kind="workflow", start_ns=i, end_ns=10, status="ok")
        for i, (id, name) in enumerate(
            [
                ("p1", "parent_one"),
                ("p2", "parent_two"),
                ("c1", "specialist"),
                ("c2", "specialist"),
            ]
        )
    ]
    run = Run(id="r", title="Separate participants", status="ok", steps=steps)
    activities = [
        {
            "step": step,
            "caller": steps[i - 2] if i >= 2 else None,
            "delegation": None,
            "model_calls": 0,
        }
        for i, step in enumerate(steps)
    ]
    diagram = collaboration_diagrams(run, {"activities": activities}, [])[0]
    assert len(diagram["lanes"]) == 4
    assert diagram["activations"][2]["x"] != diagram["activations"][3]["x"]


def test_execution_tree_defaults_to_meaningful_steps(tmp_path):
    """Simplification bypasses adapters without hiding calls or failed internals."""
    from reporting.render import execution_tree_view, tree_rows

    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")
    definitions = [
        ("root", None, "LangGraph", "workflow", "ok", {}),
        ("assess", "root", "assess", "workflow", "ok", {"langgraph_node": "assess"}),
        ("adapter", "assess", "quote_interpreter", "workflow", "ok", {}),
        ("agent", "adapter", "quote_interpreter", "workflow", "ok", {}),
        ("node", "agent", "model", "workflow", "ok", {"langgraph_node": "model"}),
        ("call", "node", "ChatOpenAI", "model", "ok", {}),
        ("failure", "agent", "internal_failure", "workflow", "error", {}),
    ]
    run = Run(
        id="r",
        title="Simplified tree",
        status="error",
        steps=[
            Step(
                id=id,
                parent_id=parent,
                name=name,
                kind=kind,
                status=status,
                context=context,
                start_ns=i,
                end_ns=20,
            )
            for i, (id, parent, name, kind, status, context) in enumerate(definitions)
        ],
    )
    rows = execution_tree_view(tree_rows(run, prices), {"agent"})
    assert [r["step"].id for r in rows if not r["detail_only"]] == [
        "agent",
        "call",
        "failure",
    ]
    assert rows[3]["summary_parent"] is None
    assert rows[5]["summary_parent"] == 3
    assert rows[5]["summary_depth"] == 1
    assert rows[3]["summary_children"]
    # Full-trace ancestry and accounting remain available without recomputation.
    assert rows[5]["parent"] == 4
    output = tmp_path / "report.html"
    render(run, prices, output)
    from html.parser import HTMLParser

    class Controls(HTMLParser):
        """Inspect actual emitted attributes, including the unchecked default."""

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("id") == "show-execution-details":
                self.checkbox = attrs
            if attrs.get("id") == "execution-tree":
                self.table = attrs

    controls = Controls()
    controls.feed(output.read_text())
    assert controls.checkbox["type"] == "checkbox"
    assert "checked" not in controls.checkbox
    assert controls.table["class"] == "summary-view"


def test_single_agent_workflow_layer_is_optional_but_real_work_is_preserved():
    """Only successful pass-through layers disappear; agent identity survives."""
    from reporting.render import execution_tree_view, tree_rows

    prices = load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")
    for status, prepare, child_kind in [
        ("ok", False, "workflow"),
        ("ok", False, "model"),
        ("ok", True, "workflow"),
        ("error", False, "workflow"),
        ("interrupted", False, "workflow"),
    ]:
        root = Step(
            id="root",
            name="workflow",
            kind="workflow",
            status=status,
            start_ns=0,
            end_ns=10,
        )
        child = Step(
            id="child",
            parent_id="root",
            name="agent_or_model",
            kind=child_kind,
            status="ok",
            start_ns=2,
            end_ns=9,
        )
        steps = [root, child]
        if prepare:
            steps.append(
                Step(
                    id="prepare",
                    parent_id="root",
                    name="prepare",
                    kind="workflow",
                    status="ok",
                    start_ns=1,
                    end_ns=2,
                    context={"langgraph_node": "prepare"},
                )
            )
        run = Run(id="r", title="Workflow containment", status="ok", steps=steps)
        scopes = {"child"} if child_kind == "workflow" else set()
        rows = execution_tree_view(tree_rows(run, prices), scopes)
        assert rows[0]["detail_only"] == (status == "ok" and not prepare)
        projected_child = next(row for row in rows if row["step"].id == "child")
        assert not projected_child["detail_only"]
        assert projected_child["summary_depth"] == (0 if rows[0]["detail_only"] else 1)
        # Full mode still exposes the original wrapper and its recorded cost.
        assert projected_child["parent"] == 0
        assert rows[0]["has_children"]
