"""Construct the shared AG-UI driver without implementing another execution loop.

Clients supply native RunAgentInput values and consume native events. This
module owns only composition and resource lifetime; LangGraphAgent alone starts
and resumes the compiled graph. A caller owns thread identity, callbacks, and
the checkpointer lifetime across runs. Missing persistence is a setup error.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from contextlib import asynccontextmanager

from ag_ui_langgraph import LangGraphAgent

from .workflow_diagram import diagram_config


@asynccontextmanager
async def open_graph(workflow):
    """Keep resources required by a graph alive throughout one run."""
    from agent_runtime.workflows.mcp_rag_chat import MCPRagWorkflow, open_agent

    # MCP's client is async and must outlive every tool call. Other workflows
    # are already compiled; opening them neither reads data nor calls a model.
    if isinstance(workflow, MCPRagWorkflow):
        async with open_agent(workflow.parameters) as graph:
            yield graph
    else:
        yield workflow


def create_langgraph_agent(graph, *, name="sample", checkpointer=None, config=None):
    """Bind caller-owned persistence; reject missing setup before execution."""
    # The conversation owns persistence across driver/graph instances. Creating
    # a saver here would hide missing setup and could lose history or approvals.
    # A supplied saver takes precedence; otherwise retain the graph's saver.
    saver = checkpointer if checkpointer is not None else graph.checkpointer
    if saver is None:
        raise ValueError(
            "create_langgraph_agent requires a checkpointer supplied by the caller "
            "or already attached to the graph"
        )
    graph.checkpointer = saver
    return LangGraphAgent(
        name=name,
        graph=graph,
        config=diagram_config(graph, {"recursion_limit": 30, **(config or {})}),
        emit_raw_events=False,
        subagent_visibility="attributed",
        emit_interrupt_outcome=True,
    )


def interaction_payload(interrupt):
    """Extract the workflow payload preserved by the official AG-UI adapter."""
    # Keep toolkit envelopes separate from domain interaction fields. Both
    # console and browser use this documented metadata shape, not tool guesses.
    return interrupt.metadata["langgraph"]["raw"]
