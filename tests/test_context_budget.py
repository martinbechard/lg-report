"""Prove shared peer compaction, child isolation, and estimated input rejection.

Scripted decisions exercise real graph, summarization, task-tool and checkpoint
behavior in sync and async paths. They do not assess live summary quality.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime.context_budget import ContextBudget, ContextBudgetExceeded
from agent_runtime.harness.simulated_model import ScriptedChatModel
from agent_runtime.tools.echo_tool import echo_tool
from agent_runtime.workflows.context_budget import build_workflow
from samples.context_budget.scripted_run import (
    CHILD_SUMMARY,
    FINAL_ANSWER,
    PEER_BRIEF,
    USER_PROMPTS,
    WORKFLOW_SUMMARY,
    make_simulated_models,
)


class Requests(BaseCallbackHandler):
    """Capture actual model inputs without inferring context from final output."""

    def __init__(self):
        self.calls = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Keep a snapshot before mutable graph state advances to another peer."""
        self.calls.append((kwargs.get("metadata", {}), str(messages)))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_peer_compaction_and_child_isolation_across_turns(asynchronous):
    """History replacement survives peer boundaries and the next checkpoint turn."""
    graph = build_workflow(*make_simulated_models())
    graph.checkpointer = InMemorySaver()
    requests = Requests()
    config = {"configurable": {"thread_id": "budget"}, "callbacks": [requests]}

    async def run_async():
        for prompt in USER_PROMPTS:
            result = await graph.ainvoke({"messages": [HumanMessage(prompt)]}, config)
        return result

    if asynchronous:
        result = asyncio.run(run_async())
    else:
        for prompt in USER_PROMPTS:
            result = graph.invoke({"messages": [HumanMessage(prompt)]}, config)

    def calls_for(description):
        return [
            text
            for metadata, text in requests.calls
            if metadata.get("report_description") == description
        ]

    planners = calls_for("Plan from shared workflow history.")
    responders = calls_for("Delegate from shared history and respond.")
    children = calls_for("Research within isolated subagent history.")
    assert len(planners) == 2 and len(responders) == 4 and len(children) == 6
    assert PEER_BRIEF in responders[0]
    # Real summary calls must occur independently for the shared workflow and
    # isolated child. A scripted final answer alone would not establish this.
    assert calls_for("Summarize shared workflow history.")
    assert calls_for("Summarize isolated subagent history.")
    assert WORKFLOW_SUMMARY in planners[1]
    assert "PARENT_ONLY_DETAIL" not in planners[1]
    assert all("PARENT_ONLY_DETAIL" not in call for call in children)
    assert all("CHILD_ONLY_DETAIL" not in call for call in responders)
    assert CHILD_SUMMARY in children[2]
    assert "CHILD_ONLY_DETAIL" not in children[2]
    assert result["messages"][-1].content == FINAL_ANSWER
    assert "PARENT_ONLY_DETAIL" not in str(result["messages"])
    assert graph.get_state(config).values["messages"] == result["messages"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_oversized_latest_input_is_rejected_before_agent_call(asynchronous):
    """An indivisible recent message cannot silently bypass the final ceiling."""
    budget = ContextBudget(max_input_tokens=250, trigger_tokens=150, keep_tokens=50)
    agent = create_agent(
        model=ScriptedChatModel(responses=[AIMessage(content="must not run")]),
        middleware=budget.middleware(
            ScriptedChatModel(responses=[AIMessage(content="summary")])
        ),
    )
    requests = Requests()
    payload = {"messages": [HumanMessage("oversized input " * 500)]}
    with pytest.raises(ContextBudgetExceeded, match="Estimated agent input"):
        if asynchronous:
            asyncio.run(agent.ainvoke(payload, {"callbacks": [requests]}))
        else:
            agent.invoke(payload, {"callbacks": [requests]})
    assert requests.calls == []


def test_budget_counts_system_instructions_too():
    """Short history is insufficient when the agent's instruction envelope is huge."""
    budget = ContextBudget(max_input_tokens=250, trigger_tokens=150, keep_tokens=50)
    agent = create_agent(
        model=ScriptedChatModel(responses=[AIMessage(content="must not run")]),
        system_prompt="lengthy instruction " * 300,
        middleware=budget.middleware(ScriptedChatModel(responses=[])),
    )
    with pytest.raises(ContextBudgetExceeded):
        agent.invoke({"messages": [HumanMessage("hello")]})


def test_budget_counts_tool_schemas_too():
    """Advertised capabilities consume input even before a tool is selected."""
    budget = ContextBudget(max_input_tokens=250, trigger_tokens=150, keep_tokens=50)
    large_tool = echo_tool.model_copy(
        update={"description": "Detailed capability description. " * 100}
    )
    agent = create_agent(
        model=ScriptedChatModel(responses=[AIMessage(content="must not run")]),
        tools=[large_tool],
        middleware=budget.middleware(ScriptedChatModel(responses=[])),
    )
    with pytest.raises(ContextBudgetExceeded):
        agent.invoke({"messages": [HumanMessage("hello")]})


@pytest.mark.parametrize("relax_workflow", [False, True])
def test_workflow_and_child_thresholds_are_independent(relax_workflow):
    """Raising one threshold suppresses only that conversation's summaries."""
    relaxed = ContextBudget(
        max_input_tokens=20000, trigger_tokens=15000, keep_tokens=5000
    )
    overrides = {"workflow_budget" if relax_workflow else "subagent_budget": relaxed}
    graph = build_workflow(*make_simulated_models(), **overrides)
    requests = Requests()
    graph.invoke(
        {"messages": [HumanMessage(USER_PROMPTS[0])]}, {"callbacks": [requests]}
    )
    descriptions = [
        metadata.get("report_description") for metadata, _ in requests.calls
    ]
    assert ("Summarize shared workflow history." in descriptions) is not relax_workflow
    assert ("Summarize isolated subagent history." in descriptions) is relax_workflow


@pytest.mark.parametrize(
    "values", [(100, 100, 10), (100, 20, 30), (100, 20, 0), (True, 20, 10)]
)
def test_invalid_budget_is_rejected_at_configuration(values):
    """Fail incoherent policy before a provider or graph can be invoked."""
    with pytest.raises(ValueError):
        ContextBudget(*values)
