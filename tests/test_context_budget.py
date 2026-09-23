"""Prove shared peer compaction, child isolation, and estimated input rejection.

Scripted decisions exercise real graph, summarization, task-tool and checkpoint
behavior in sync and async paths. They do not assess live summary quality.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime.context_budget import (
    ContextBudget,
    ContextBudgetExceeded,
    InputBudgetMiddleware,
    context_estimate,
)
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
        self.compactions = []

    def on_custom_event(self, name, data, **kwargs):
        """Observe completed replacements independently of model responses."""
        if name == "context_compaction":
            self.compactions.append(data)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Keep a snapshot before mutable graph state advances to another peer."""
        self.calls.append((kwargs.get("metadata", {}), str(messages)))


def test_usage_baseline_controls_trigger_even_when_character_estimate_disagrees():
    """Use a measured receipt, not a larger character guess or cached-token sum.

    An intentionally exaggerated string makes the two counters disagree. The
    usage receipt includes 60 cached input tokens and 20 reasoning output tokens;
    both are already included in their parent totals and must not be added twice.
    """
    history = [HumanMessage("background " * 100)]
    answer = AIMessage(content="answer", usage_metadata={
        "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        "input_token_details": {"cache_read": 60},
        "output_token_details": {"reasoning": 20},
    })
    request = SimpleNamespace(messages=history, system_message=SystemMessage("role"), tools=[])
    InputBudgetMiddleware(1000)._record_receipt(request, SimpleNamespace(result=[answer]))
    retained = [*history, answer]
    middleware = ContextBudget(1000, 200, 40).middleware(ScriptedChatModel(responses=[]))[0]
    assert count_tokens_approximately(retained) > 200
    assert context_estimate(retained) == (150, "reported input + output")
    assert not middleware._should_summarize(retained, 999)

    # Actual new observations were not in the provider's receipt. Estimate only
    # this suffix, rather than recounting the entire history with characters/4.
    observation = ToolMessage(content="observation " * 30, tool_call_id="lookup")
    expected = 150 + count_tokens_approximately([observation])
    assert context_estimate([*retained, observation])[0] == expected
    assert middleware._should_summarize([*retained, observation], 0)


def test_replacement_invalidates_usage_but_graph_ids_do_not():
    """A retained assistant receipt must never resurrect compacted-away tokens."""
    history = [HumanMessage("original background")]
    answer = AIMessage(content="answer", usage_metadata={
        "input_tokens": 900, "output_tokens": 50, "total_tokens": 950,
    })
    request = SimpleNamespace(messages=history, system_message=SystemMessage("role"), tools=[])
    InputBudgetMiddleware(2000)._record_receipt(request, SimpleNamespace(result=[answer]))
    history[0].id = "assigned-by-reducer"
    assert context_estimate([*history, answer])[0] == 950
    replaced = [HumanMessage("summary"), answer]
    count, basis = context_estimate(replaced)
    assert count == count_tokens_approximately(replaced) + count_tokens_approximately([request.system_message])
    assert basis == "local estimate after history replacement"
    middleware = ContextBudget(1000, 200, 40).middleware(ScriptedChatModel(responses=[]))[0]
    assert not middleware._should_summarize(replaced, 950)


def test_input_guard_reuses_usage_and_adjusts_changed_peer_instructions():
    """Do not reject measured small context because an encoded block looks large.

    A role switch must still account for new instructions. This distinguishes
    legitimate reuse of measured history from simply disabling the input guard.
    """
    history = [HumanMessage("hello")]
    answer = AIMessage(content=[{"type": "reasoning", "encrypted_content": "x" * 12000}],
                       usage_metadata={"input_tokens": 100, "output_tokens": 50,
                                       "total_tokens": 150})
    request = SimpleNamespace(messages=history, system_message=SystemMessage("short role"), tools=[])
    guard = InputBudgetMiddleware(1000)
    guard._record_receipt(request, SimpleNamespace(result=[answer]))
    request.messages = [*history, answer]
    assert count_tokens_approximately(request.messages) > 1000
    guard.check(request)
    request.system_message = SystemMessage("long new instructions " * 300)
    with pytest.raises(ContextBudgetExceeded):
        guard.check(request)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_compaction_counts_match_replaced_history_and_failure_is_not_completion(
    asynchronous, monkeypatch
):
    """Compare emitted counts to real native output, then force summary failure."""
    budget = ContextBudget(max_input_tokens=1000, trigger_tokens=150, keep_tokens=40)
    middleware = budget.middleware(
        ScriptedChatModel(responses=[AIMessage(content="A short summary.")])
    )[0]
    messages = [HumanMessage("Older background. " * 100), AIMessage("Noted."),
                HumanMessage("Continue.")]
    requests = Requests()

    def run(_):
        """Supply a native callback scope for synchronous custom events."""
        return middleware.before_model({"messages": messages}, None)

    async def arun(_):
        """Supply the equivalent scope for asynchronous custom events."""
        return await middleware.abefore_model({"messages": messages}, None)

    runnable = RunnableLambda(run, afunc=arun)
    config = {"callbacks": [requests]}
    result = (asyncio.run(runnable.ainvoke({}, config)) if asynchronous
              else runnable.invoke({}, config))
    assert len(requests.compactions) == 1
    event = requests.compactions[0]
    assert event["compaction_before_tokens"] == count_tokens_approximately(messages)
    retained = result["messages"][1:]
    assert event["compaction_after_tokens"] == count_tokens_approximately(retained)
    assert event["compaction_after_messages"] == len(retained)
    assert event["compaction_keep_tokens"] == 40
    assert event["compaction_max_input_tokens"] == 1000

    def fail(*args):
        """Fail the summary boundary without exercising provider retry delays."""
        raise RuntimeError("summary unavailable")

    async def afail(*args):
        """Fail the asynchronous summary boundary identically."""
        fail()

    monkeypatch.setattr(middleware, "_create_summary", fail)
    monkeypatch.setattr(middleware, "_acreate_summary", afail)
    with pytest.raises(RuntimeError, match="summary unavailable"):
        if asynchronous:
            asyncio.run(runnable.ainvoke({}, config))
        else:
            runnable.invoke({}, config)
    assert len(requests.compactions) == 1


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
    # The last complete tool pair is retained even when it exceeds keep_tokens.
    assert "CHILD_ONLY_DETAIL" in children[2]
    assert result["messages"][-1].content == FINAL_ANSWER
    assert "context_budget_receipt" in result["messages"][-1].response_metadata
    assert "PARENT_ONLY_DETAIL" not in str(result["messages"])
    assert graph.get_state(config).values["messages"] == result["messages"]
    # Both lifecycle paths publish count-only events after native replacement.
    # Three fixture summaries imply three successful compactions, not one event
    # for every before-model check or every ordinary model invocation.
    assert len(requests.compactions) == 3
    assert {c["compaction_trigger_tokens"] for c in requests.compactions} == {1500, 1200}
    # A replacement can grow when a summary plus its framing is longer than
    # the removed messages; reporting must retain that evidence honestly.
    assert all(c["compaction_before_tokens"] > 0 and c["compaction_after_tokens"] > 0
               for c in requests.compactions)
    assert all(c["compaction_event"] == "completed" for c in requests.compactions)


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
    assert requests.compactions == []


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
