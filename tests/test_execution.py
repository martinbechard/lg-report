"""Verify that driver construction preserves conversation-owned persistence.

A driver must never silently replace missing persistence with a fresh saver:
that would lose history and pending approvals across graph openings.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_runtime.harness.execution import create_langgraph_agent


def make_graph(checkpointer=None):
    """Compile a minimal graph to test real adapter wiring without model calls."""
    builder = StateGraph(MessagesState)
    builder.add_edge(START, END)
    return builder.compile(checkpointer=checkpointer)


def test_driver_requires_conversation_persistence():
    """Fail before execution when neither caller nor graph supplies persistence."""
    graph = make_graph()
    with pytest.raises(ValueError, match="checkpointer"):
        create_langgraph_agent(graph)
    assert graph.checkpointer is None


@pytest.mark.parametrize("explicit", [False, True])
def test_driver_reuses_conversation_persistence(explicit):
    """Repeated drivers keep the exact supplied or graph-attached saver."""
    saver = InMemorySaver()
    graph = make_graph(InMemorySaver() if explicit else saver)
    kwargs = {"checkpointer": saver} if explicit else {}
    create_langgraph_agent(graph, **kwargs)
    assert graph.checkpointer is saver
    create_langgraph_agent(graph)
    assert graph.checkpointer is saver
