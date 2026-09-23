"""Verify the MCP sample's reporting bridge without downloads or paid models.

An in-process FastMCP fixture exercises protocol adaptation and real Deep Agent
routing. Separate server tests and the standalone smoke run cover stdio transport.

AI attribution: Modified with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from fixtures.mock_client import MockClient
from langchain.mcp import MCPAdapter

from agent_runtime.harness.conversation import Conversation, Request
from agent_runtime.harness.sample_catalog import SampleCatalog
from agent_runtime.mcp_servers.wikipedia import build_server
from agent_runtime.workflows import mcp_rag_chat
from reporting.execute_runnable import execute_runnable
from reporting.pricing import load_prices
from reporting.schema import Run

create_run = partial(SampleCatalog().create_run, "mcp_rag_chat")
from samples.mcp_rag_chat.scripted_run import USER_PROMPTS


class Collection:
    """Return one labelled passage so reports can prove evidence propagation."""

    def count(self):
        # Advertise one available passage so the server will attempt retrieval.
        return 1

    def query(self, **kwargs):
        # Supply recognizable evidence for protocol/report propagation checks.
        # Ignore Chroma query kwargs: these tests check transport and result handling,
        # not ranking. Nested lists represent one query with one returned passage.
        return {
            "ids": [["fixture-raven"]],
            "documents": [["The raven adapts to urban habitats."]],
            "metadatas": [[{"source": "fixture", "title": "Raven"}]],
            "distances": [[0.1]],
        }


# Replace only the agent lifetime boundary with a deterministic local
# server, then verify evidence, accounting, and cleanup reach the report.
def test_sample_records_mcp_evidence(tmp_path, monkeypatch):
    lifecycle = []

    @asynccontextmanager
    async def open_fixture(parameters):
        # Keep an MCP session alive while the synchronous bridge uses its agent.
        # `model` is the scripted model passed by the workflow. Yield a real Deep
        # Agent configured with discovered tools; finally records context exit even
        # when the caller raises. The transport is in-process, not stdio.
        async with MCPAdapter(build_server(Collection())) as adapter:
            lifecycle.append("opened")
            try:
                yield create_deep_agent(
                    **parameters,
                    tools=await adapter.list_tools(),
                )
            finally:
                lifecycle.append("closed")

    monkeypatch.setattr(mcp_rag_chat, "open_agent", open_fixture)
    workflow, provider, model = create_run(False)
    client = MockClient([Request(USER_PROMPTS[0])])
    execute_runnable(
        Conversation(workflow, client),
        {},
        tmp_path / "report",
        load_prices(Path(__file__).parents[1] / "models.json"),
        provider=provider,
        model=model,
        demo=True,
        include_output=True,
    )
    run = Run.model_validate_json((tmp_path / "report/run.json").read_text())
    assert run.status == "ok" and run.demo
    models = sorted(
        [s for s in run.steps if s.kind == "model"], key=lambda s: s.start_ns
    )
    # These are recorded executions, not the model proposed tool_calls. The
    # second model request must include evidence returned by the MCP tool.
    tools = [s for s in run.steps if s.kind == "tool"]
    assert len(models) == 2
    assert len(tools) == 1 and tools[0].name == "semantic_search_wikipedia"
    assert "fixture-raven" in json.dumps(models[1].request)
    assert all(step.usage.input_tokens > 0 for step in models)
    assert (tmp_path / "report/report.html").is_file()
    assert lifecycle == ["opened", "closed"]


# A failed agent call must still close the async MCP-backed lifetime;
# this guards resource cleanup independently from successful reporting.
def test_bridge_closes_on_failure(monkeypatch):
    lifecycle = []

    class BrokenAgent:
        async def ainvoke(self, inputs, config):
            # Fail after checking that the bridge forwards history and tracing config.
            # This async replacement returns no result; the surrounding context manager
            # must close when the exception propagates.
            assert config["metadata"]["report_turn"] == 2
            assert inputs["messages"] == ["prior history"]
            raise ValueError("agent failed")

    @asynccontextmanager
    async def open_fixture(parameters):
        # Exercise the bridge cleanup path with an agent that always fails.
        # Accept the workflow model argument for interface compatibility; yield the
        # failure stub and record cleanup when the async context is exited.
        try:
            yield BrokenAgent()
        finally:
            lifecycle.append("closed")

    monkeypatch.setattr(mcp_rag_chat, "open_agent", open_fixture)
    workflow, _, _ = create_run(False)
    with pytest.raises(ValueError, match="agent failed"):
        workflow.invoke(
            {"messages": ["prior history"]}, {"metadata": {"report_turn": 2}}
        )
    assert lifecycle == ["closed"]
