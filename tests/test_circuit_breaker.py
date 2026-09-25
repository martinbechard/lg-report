"""Verify that real limits stop rejected writes without creating an artifact.

Scripted decisions remove provider variability; sync and async checks exercise
native middleware termination, tool-result pairing, and the actual backend.
The filename contract check prevents the working sample drifting into the
deliberately broken assignment used by this demonstration.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_runtime.agents import planner
from agent_runtime.harness.model_factory import build_model, model_factory_scope
from agent_runtime.harness.sample_catalog import SampleCatalog
from agent_runtime.harness.simulated_model import SimulatedModel
from agent_runtime.workflows.circuit_breaker import build_workflow
from agent_runtime.workflows.exercise_backend import ExerciseBackend
from samples.circuit_breaker.sample import CONVERSATION


@pytest.mark.parametrize("asynchronous", [False, True])
def test_breaker_blocks_fourth_write_and_ends(tmp_path, asynchronous):
    """The final message must come from middleware, not a scripted answer."""
    # Exercise the same conversation extraction used by the sample catalog.
    with model_factory_scope(live=False, conversation=CONVERSATION):
        model = build_model(caller="worker")
    graph = build_workflow(model=model, workspace_dir=tmp_path)
    prompt = SampleCatalog().prompts("circuit_breaker")[0]
    state = {"messages": [HumanMessage(prompt)]}
    result = asyncio.run(graph.ainvoke(state)) if asynchronous else graph.invoke(state)
    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(replies) == 4
    assert [m.tool_call_id for m in replies] == [f"forbidden-write-{i}" for i in range(1, 5)]
    assert all("Only /plan.md, /slug.py, and /test_slug.py may be written" in m.content for m in replies[:3])
    assert replies[-1].status == "error"
    assert "limit" in replies[-1].content.lower()
    assert "limit" in result["messages"][-1].text.lower()
    assert not list(tmp_path.iterdir())


def test_model_limit_stops_reads_that_do_not_consume_write_allowance(tmp_path):
    """The fallback ceiling also bounds a different kind of unproductive loop."""
    model = SimulatedModel(conversation=[{"role": "test-agent", "content": response.content, "tool_calls": response.tool_calls, "response_metadata": response.response_metadata} for response in [
        AIMessage(content="", tool_calls=[{
            "name": "read_file", "args": {"file_path": "/slugify.py"}, "id": f"read-{i}",
        }]) for i in range(8)
    ]], agent_name="test-agent")
    result = build_workflow(model=model, workspace_dir=tmp_path).invoke(
        {"messages": [HumanMessage("Inspect the required file.")]}
    )
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 6
    assert "Model call limits exceeded" in result["messages"][-1].text
    assert not list(tmp_path.iterdir())


def test_working_planner_names_files_allowed_by_backend(monkeypatch, tmp_path):
    """Check the live role instruction against real backend capabilities."""
    monkeypatch.setattr(planner, "create_agent", lambda **kwargs: SimpleNamespace(**kwargs))
    prompt = planner.build_agent({}, "").system_prompt
    backend = ExerciseBackend(tmp_path)
    for path in ("/slug.py", "/test_slug.py"):
        assert path in prompt
        assert backend.write(path, "# contract check").error is None
    assert "from slug" in prompt
    assert "/slugify.py" not in prompt
    assert backend.write("/slugify.py", "# rejected").error
