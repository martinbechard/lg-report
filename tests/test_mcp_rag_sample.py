"""Verify the MCP sample's reporting bridge without downloads or paid models.

An in-process FastMCP fixture exercises protocol adaptation and real Deep Agent
routing. Separate server tests and the standalone smoke run cover stdio transport.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.mcp import MCPAdapter

from lg_report.mcp_servers.wikipedia import build_server
from lg_report.platform.conversation import Conversation, Request
from lg_report.platform.static_client import StaticClient
from lg_report.report.pricing import load_prices
from lg_report.report.recording import record_run
from lg_report.report.schema import Run
from lg_report.workflows import mcp_rag_chat
from samples.mcp_rag_chat.app import create_run
from samples.mcp_rag_chat.test_case import USER_PROMPTS


class Collection:
    """Return one labelled passage so reports can prove evidence propagation."""

    def count(self):
        return 1

    def query(self, **kwargs):
        return {
            "ids": [["fixture-raven"]],
            "documents": [["The raven adapts to urban habitats."]],
            "metadatas": [[{"source": "fixture", "title": "Raven"}]],
            "distances": [[0.1]],
        }


def test_sample_records_mcp_evidence(tmp_path, monkeypatch):
    lifecycle = []

    @asynccontextmanager
    async def open_fixture(model):
        async with MCPAdapter(build_server(Collection())) as adapter:
            lifecycle.append("opened")
            try:
                yield create_deep_agent(
                    model=model,
                    tools=await adapter.list_tools(),
                    backend=StateBackend(),
                )
            finally:
                lifecycle.append("closed")

    monkeypatch.setattr(mcp_rag_chat, "open_agent", open_fixture)
    workflow, provider, model = create_run(False)
    client = StaticClient([Request(USER_PROMPTS[0])])
    record_run(
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
    tools = [s for s in run.steps if s.kind == "tool"]
    assert len(models) == 2
    assert len(tools) == 1 and tools[0].name == "search_wikipedia"
    assert "fixture-raven" in json.dumps(models[1].request)
    assert all(step.usage.input_tokens > 0 for step in models)
    assert (tmp_path / "report/report.html").is_file()
    assert lifecycle == ["opened", "closed"]


def test_bridge_closes_on_failure(monkeypatch):
    lifecycle = []

    class BrokenAgent:
        async def ainvoke(self, inputs, config):
            assert config["metadata"]["report_turn"] == 2
            assert inputs["messages"] == ["prior history"]
            raise ValueError("agent failed")

    @asynccontextmanager
    async def open_fixture(model):
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
