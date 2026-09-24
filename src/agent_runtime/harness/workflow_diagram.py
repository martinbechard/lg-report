"""Snapshot declared workflow states for portable, offline report diagrams.

Compiled graphs expose their construction-time nodes, edges and branch maps.
Capture only that topology, never executable objects or application state. Native
subgraphs are discovered by LangGraph. Opaque wrapper calls may expose child
graph references, but no workflow supplies a hand-maintained node/edge diagram.
This describes possible transitions, not proof that every branch ran.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import inspect
import json

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode


def diagram_config(graph, config=None):
    """Snapshot the graph at execution time without modifying caller configuration.

    Both client drivers and direct runnable recording use this boundary. Objects
    without a compiled graph (for example a conversation loop) are left alone;
    their eventual driver captures the actual graph after resources are opened.
    """
    result = dict(config or {})
    if isinstance(graph, CompiledStateGraph):
        result["metadata"] = {
            **result.get("metadata", {}),
            "report_workflow_definition": json.dumps(workflow_definition(graph)),
        }
    return result


def workflow_definition(graph):
    """Project a compiled StateGraph and its child graph references.

    Branch maps preserve multiple decisions that reach the same destination;
    the standard drawable graph can collapse those labels. Unmapped
    conditional routes are identified as unresolved rather than guessed. Graph
    references for opaque wrappers are composition metadata, never copied edges.
    """
    builder = graph.builder
    edges = [
        {"source": source, "target": target, "label": ""}
        for source, target in sorted(builder.edges)
    ]
    unresolved = []
    for sources, target in sorted(builder.waiting_edges):
        edges.extend(
            {"source": source, "target": target, "label": "join: all predecessors"}
            for source in sources
        )
    for source, branches in builder.branches.items():
        for branch in branches.values():
            if branch.ends is None:
                unresolved.append(source)
                continue
            edges.extend(
                {"source": source, "target": target, "label": str(label)}
                for label, target in branch.ends.items()
            )
    # Command destinations belong to the native node definition rather than a
    # conditional branch. Preserve them when provided by the graph author.
    for source, node in builder.nodes.items():
        destinations = node.ends or ()
        edges.extend(
            {
                "source": source,
                "target": target,
                "label": str(destinations[target])
                if isinstance(destinations, dict)
                else "command",
            }
            for target in destinations
        )
    subgraphs = dict(graph.get_subgraphs())
    subgraphs.update(getattr(graph, "report_subgraphs", {}))
    children = {
        name: workflow_definition(child)
        for name, child in subgraphs.items()
        if isinstance(child, CompiledStateGraph)
    }
    delegations = []
    for source, node in builder.nodes.items():
        # DeepAgents keeps the actual compiled delegates in its task closure.
        # This narrow adapter reads only that registry, never arbitrary closure
        # data or tool descriptions. Native subgraph discovery omits tool calls.
        if not isinstance(node.runnable, ToolNode):
            continue
        task = node.runnable.tools_by_name.get("task")
        function = getattr(task, "func", None)
        if (
            not inspect.isfunction(function)
            or function.__module__ != "deepagents.middleware.subagents"
        ):
            continue
        registry = inspect.getclosurevars(function).nonlocals.get("subagent_graphs")
        if not isinstance(registry, dict):
            raise TypeError(
                "DeepAgents task registry changed; delegation capture needs updating"
            )
        for name, delegate in registry.items():
            if isinstance(delegate, CompiledStateGraph):
                delegate_definition = workflow_definition(delegate)
                comment = getattr(graph, "report_delegate_comments", {}).get(name)
                if comment:
                    delegate_definition["context"] = comment
                    delegate_definition["annotations"]["model"]["comment"] = comment
                delegations.append(
                    {
                        "source": source,
                        "tool": "task",
                        "name": name,
                        "graph": delegate_definition,
                    }
                )
    if not children.keys() <= builder.nodes.keys():
        raise ValueError("A child workflow must belong to a declared call node")
    # Traverse declared edges from START so registration order cannot put an
    # initialization state below its successors. Loops visit each state once.
    ordered, pending = [], ["__start__"]
    while pending:
        node = pending.pop(0)
        if node in ordered or node == "__end__":
            continue
        ordered.append(node)
        pending.extend(edge["target"] for edge in edges if edge["source"] == node)
    ordered.extend(node for node in builder.nodes if node not in ordered)
    description = getattr(
        graph, "report_context", "Compiled graph; arrows show declared transitions"
    )
    if unresolved:
        description += ". Dynamic destinations unavailable for: " + ", ".join(
            unresolved
        )
    # These short code-owned comments describe states without duplicating routes.
    # Native tool registries identify available operations; no model is invoked.
    annotations = {}
    for name, node in builder.nodes.items():
        metadata = node.metadata or {}
        comment = metadata.get("report_comment", "")
        kind = "process"
        if name in children:
            kind = "subgraph"
        elif isinstance(node.runnable, ToolNode):
            kind = "tool"
            comment = comment or "Run tools; append results to history. Available: " + ", ".join(node.runnable.tools_by_name)
        elif name == "model":
            kind = "model"
            comment = getattr(graph, "report_model_comment", "Choose a tool or return an answer")
        elif name == "SummarizationMiddleware.before_model":
            # This hook belongs to the invocation runtime, not the role prompt.
            kind = "middleware"
            comment = "Before each model call: compact history if needed"
        annotations[name] = {"kind": kind, "comment": comment}
    return {
        "annotations": annotations,
        "title": getattr(graph, "report_title", graph.name),
        "context": description,
        "nodes": [*ordered, "__end__"],
        "edges": edges,
        "children": children,
        "delegations": delegations,
    }
