"""Protect portable state diagrams from invented topology and unsafe labels.

These tests use saved definitions rather than executing models. The sample's
CLI integration tests separately verify real graph capture and section ordering.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from types import SimpleNamespace
from xml.etree import ElementTree

from reporting.workflow_diagram import workflow_diagrams


def test_missing_topology_has_no_invented_diagram():
    """Old traces remain readable without guessing states from agent order."""
    assert workflow_diagrams(SimpleNamespace(steps=[SimpleNamespace(context={})])) == []


def test_saved_labels_are_escaped_and_repeated_topology_is_deduplicated():
    """Report markup must treat graph names and branch labels as inert text."""
    payload = '<script>alert("diagram")</script>'
    definition = {
        "title": payload,
        "context": "No shared state",
        "children": {},
        "nodes": ["__start__", payload, "__end__"],
        "edges": [
            {"source": "__start__", "target": payload, "label": payload},
            {"source": payload, "target": "__end__", "label": ""},
        ],
    }
    step = SimpleNamespace(
        context={"report_workflow_definition": json.dumps(definition)}
    )
    diagrams = workflow_diagrams(SimpleNamespace(steps=[step, step]))
    assert len(diagrams) == 1
    svg = ElementTree.fromstring(diagrams[0])
    assert all(not element.tag.endswith("script") for element in svg.iter())
    assert payload in "".join(svg.itertext())


def test_topology_is_generated_from_compiled_graph_without_sample_metadata():
    """Untaken branches and later graph changes appear without editing a diagram."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    from agent_runtime.harness.workflow_diagram import diagram_config

    builder = StateGraph(MessagesState)
    builder.add_node("decide", lambda state: {})
    builder.add_node("retry", lambda state: {})
    builder.add_edge(START, "decide")
    builder.add_conditional_edges(
        "decide", lambda state: "done", {"done": END, "again": "retry", "cancel": END}
    )
    builder.add_edge("retry", "decide")
    original = {"metadata": {"caller": "test"}}
    config = diagram_config(builder.compile(), original)
    definition = json.loads(config["metadata"]["report_workflow_definition"])
    assert original == {"metadata": {"caller": "test"}}
    assert {e["label"] for e in definition["edges"] if e["source"] == "decide"} == {
        "done",
        "again",
        "cancel",
    }
    builder.add_node("extra", lambda state: {})
    builder.add_edge("retry", "extra")
    builder.add_edge("extra", END)
    updated = json.loads(
        diagram_config(builder.compile())["metadata"]["report_workflow_definition"]
    )
    assert "extra" in updated["nodes"]
    assert "extra" not in definition["nodes"]


def test_native_children_and_unmapped_routes_are_truthful():
    """Discover real nested graphs and label unknown destinations without guessing."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    from agent_runtime.harness.workflow_diagram import workflow_definition

    child = StateGraph(MessagesState)
    child.add_node("call", lambda state: {})
    child.add_edge(START, "call")
    child.add_edge("call", END)
    parent = StateGraph(MessagesState)
    parent.add_node("nested", child.compile())
    parent.add_edge(START, "nested")
    parent.add_conditional_edges("nested", lambda state: END)
    definition = workflow_definition(parent.compile())
    assert definition["children"]["nested"]["nodes"] == [START, "call", END]
    assert "Dynamic destinations unavailable for: nested" in definition["context"]
    assert definition["edges"] == [{"source": START, "target": "nested", "label": ""}]


def test_repeated_child_flow_shares_box_but_different_steps_do_not():
    """Two call sites share a definition, while a changed flow keeps its own box."""
    from copy import deepcopy

    child = {
        "title": "planner",
        "context": "Shared history",
        "children": {},
        "nodes": ["__start__", "model", "__end__"],
        "edges": [
            {"source": "__start__", "target": "model", "label": ""},
            {"source": "model", "target": "__end__", "label": ""},
        ],
    }
    parent = {
        "title": "workflow",
        "context": "",
        "nodes": ["plan", "update"],
        "edges": [{"source": "plan", "target": "update", "label": ""}],
        "children": {"plan": child, "update": deepcopy(child)},
    }

    def inspect():
        """Read rendered boxes and call links without depending on layout sizes."""
        step = SimpleNamespace(
            context={"report_workflow_definition": json.dumps(parent)}
        )
        svg = ElementTree.fromstring(
            workflow_diagrams(SimpleNamespace(steps=[step]))[0]
        )
        titles = [
            node.text
            for node in svg.iter()
            if node.tag.endswith("text") and node.text == "planner"
        ]
        links = [
            node.attrib["d"]
            for node in svg.iter()
            if node.attrib.get("stroke-dasharray")
        ]
        return titles, links

    titles, links = inspect()
    assert len(titles) == 1 and len(links) == 2
    assert links[0].split(" V ")[1] == links[1].split(" V ")[1]
    # A new step with the same agent name must not reuse the old flow's box.
    changed = parent["children"]["update"]
    changed["nodes"].insert(2, "edit")
    changed["edges"][1]["target"] = "edit"
    changed["edges"].append({"source": "edit", "target": "__end__", "label": ""})
    titles, links = inspect()
    assert len(titles) == 2
    assert links[0].split(" V ")[1] != links[1].split(" V ")[1]


def test_context_budget_discovers_review_and_repair_edges():
    """Read the actual review graph and response routes without running models."""
    from agent_runtime.harness.sample_catalog import SampleCatalog
    from agent_runtime.harness.workflow_diagram import workflow_definition

    graph, *_ = SampleCatalog().create_run("context_budget", False, tracing=False)
    try:
        definition = workflow_definition(graph)
        assert (
            definition["children"]["planning_step"]
            == definition["children"]["plan_update_step"]
        )
        assert definition["annotations"]["planning_step"]["kind"] == "subgraph"
        assert "Prepare the file plan" in definition["annotations"]["planning_step"]["comment"]
        worker = definition["children"]["working_step"]
        assert worker["annotations"]["model"]["kind"] == "model"
        assert worker["annotations"]["tools"]["kind"] == "tool"
        assert worker["annotations"]["SummarizationMiddleware.before_model"]["kind"] == "middleware"
        assert worker["delegations"] == []
        assert definition['children']['review_step']['title'] == 'isolated-reviewer'
        routes = {(e['source'], e['target'], e['label']) for e in definition['edges']}
        assert ('review_step', 'working_step', 'repair') in routes
        assert ('review_step', 'plan_update_step', 'approved') in routes
        assert ('plan_update_step', 'working_step', 'work') in routes
        step = SimpleNamespace(
            context={"report_workflow_definition": json.dumps(definition)}
        )
        svg = ElementTree.fromstring(
            workflow_diagrams(SimpleNamespace(steps=[step]))[0]
        )
        assert sum(node.text == "planner" for node in svg.iter()) == 1
        assert any(node.text == "isolated-reviewer" for node in svg.iter())
        assert not any("marker-start" in node.attrib for node in svg.iter())
        assert any(node.text == "approved" for node in svg.iter())
        assert any(node.text == "Runtime middleware" for node in svg.iter())
        assert any(node.text == "Agent behavior" for node in svg.iter())
        assert not any(node.text == "Context budget?" for node in svg.iter())
        assert any(
            node.attrib.get("data-source") == "tools"
            and node.attrib.get("data-target") == "SummarizationMiddleware.before_model"
            for node in svg.iter()
        )
        assert "append results to history" in worker["annotations"]["tools"]["comment"]
        # Rendering must preserve the captured hook and routes, even though its
        # ownership is drawn separately from the model/tool behavior.
        assert "SummarizationMiddleware.before_model" in worker["nodes"]
        assert any(edge["source"] == "SummarizationMiddleware.before_model"
                   and edge["target"] == "model" for edge in worker["edges"])
        assert {node.attrib.get("data-node-kind") for node in svg.iter()} >= {
            "model", "tool", "middleware", "subgraph"
        }
    finally:
        graph.workspace.cleanup()
