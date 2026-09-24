"""Verify one scenario supplies named agents without prompt or tool inference.

Real native graphs and streaming calls exercise metadata propagation. Per-agent
cursors and cache ledgers must stay independent when the model object is shared.
AI assistance: Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from agent_runtime.harness.simulated_model import SimulatedModel


def scenario():
    """Interleave client/tool evidence with two agents using identical capabilities."""
    return [
        {"role": "client", "content": "Start"},
        {"role": "alpha", "content": "Alpha one"},
        {"role": "tool", "content": "Expected observation, never a model answer"},
        {"role": "beta", "content": "Beta one"},
        {"role": "alpha", "content": "Alpha two"},
        {"role": "human", "content": "Approve"},
        {"role": "beta", "content": "Beta two"},
    ]


def test_shared_native_agents_select_by_name_with_identical_prompts():
    """Request order may differ from story order; only each agent's order is fixed."""
    story = scenario()
    model = SimulatedModel(conversation=story)
    agents = {
        name: create_agent(model, name=name, system_prompt="Same instructions")
        for name in ("alpha", "beta")
    }
    histories = {}
    for name in ("beta", "alpha"):
        histories[name] = agents[name].invoke({"messages": [("user", "Start")]})[
            "messages"
        ]
        response = histories[name][-1]
        assert response.content == f"{name.title()} one"
        assert response.usage_metadata["input_token_details"]["cache_read"] == 0
    for name in ("alpha", "beta"):
        response = agents[name].invoke(
            {"messages": [*histories[name], HumanMessage("Continue")]}
        )["messages"][-1]
        assert response.content == f"{name.title()} two"
        assert response.usage_metadata["input_token_details"]["cache_read"] > 0
    assert story == scenario()
    assert model._positions == {"alpha": 2, "beta": 2}


def test_async_stream_preserves_identity_and_usage_once():
    """Native asynchronous model events must consume one entry and one usage record."""
    model = SimulatedModel(conversation=scenario())
    agent = create_agent(model, name="alpha")

    async def collect():
        return [
            event["data"]["chunk"]
            async for event in agent.astream_events(
                {"messages": [("user", "Start")]}, version="v2"
            )
            if event["event"] == "on_chat_model_stream"
        ]

    chunks = asyncio.run(collect())
    assert sum(chunk.usage_metadata is not None for chunk in chunks) == 1
    assert "".join(chunk.content for chunk in chunks) == "Alpha one"
    assert model._positions == {"alpha": 1}


def test_missing_unknown_and_exhausted_identity_fail_explicitly():
    """Never borrow another role's answer or cycle back to a completed response."""
    model = SimulatedModel(conversation=[{"role": "alpha", "content": "Only answer"}])
    with pytest.raises(ValueError, match="named agent"):
        model.invoke("Start")
    with pytest.raises(ValueError, match="unknown"):
        model.invoke("Start", config={"metadata": {"lc_agent_name": "unknown"}})
    config = {"metadata": {"lc_agent_name": "alpha"}}
    answer = model.invoke("Start", config=config)
    with pytest.raises(ValueError, match="No simulated response left"):
        model.invoke(
            [HumanMessage("Start"), answer, HumanMessage("Continue")], config=config
        )
    assert model._positions == {"alpha": 1}


def test_concurrent_roles_and_multiple_tools_keep_bindings_isolated():
    """Shared state cannot mix identities even when bindings have identical tools."""

    def first(query: str) -> str:
        """Return input for binding inspection; this test does not execute the tool."""
        return query

    def second(query: str) -> str:
        """Provide a second schema, which must not be treated as an agent selector."""
        return query

    model = SimulatedModel(conversation=scenario())
    binding = model.bind_tools([first, second])
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(
            pool.map(
                lambda name: binding.invoke(
                    "Start", config={"metadata": {"lc_agent_name": name}}
                ),
                ("alpha", "beta"),
            )
        )
    assert [answer.content for answer in answers] == ["Alpha one", "Beta one"]
    assert all(
        answer.usage_metadata["input_token_details"]["cache_read"] == 0
        for answer in answers
    )
    assert len(binding.kwargs["tool_definitions"]) == 2


def test_failed_metering_does_not_consume_response_and_runs_are_isolated():
    """A rewritten prefix fails without skipping a reply or mutating the scenario."""
    story = scenario()
    model = SimulatedModel(conversation=story, agent_name="alpha")
    answer = model.invoke("Start")
    with pytest.raises(ValueError, match="previous context"):
        model.invoke("Replaced history")
    response = model.invoke([HumanMessage("Start"), answer, HumanMessage("Continue")])
    assert response.content == "Alpha two"
    assert (
        SimulatedModel(conversation=story, agent_name="alpha").invoke("Start").content
        == "Alpha one"
    )
