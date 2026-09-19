"""Verify MCP discovery, retrieval contracts, and Deep Agent tool execution.

A tiny real Chroma collection uses deterministic embeddings without downloads.
A child Python server verifies stdio independently of the in-process MCP tests.
The scripted model verifies routing, not live model answer quality.

AI attribution: Modified with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import sys
from pathlib import Path

import chromadb
import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage
from test_rag import TestEmbedding

from lg_report.mcp_servers.wikipedia import build_server
from lg_report.platform.simulated_model import MeteredDemoModel


@pytest.fixture
def collection(tmp_path):
    # Keep retrieval deterministic and local while exercising the real
    # persistent Chroma collection boundary used by the MCP server.
    collection = chromadb.PersistentClient(path=str(tmp_path / "db")).create_collection(
        "mcp-test", embedding_function=TestEmbedding()
    )
    collection.add(
        ids=[f"raven-{i}" for i in range(6)],
        documents=["A raven adapts to urban environments."] * 6,
        metadatas=[{"title": "Raven", "source": "fixture"}] * 6,
    )
    return collection


# Cover tool discovery, input limits, retrieval payload shape, and
# Deep Agent tool routing in one protocol-level offline scenario.
def test_mcp_contract_and_deep_agent(collection):
    async def exercise():
        # Run async discovery, retrieval, and agent execution in one adapter lifetime.
        # Use the enclosing local collection; asyncio.run below executes this coroutine
        # from the synchronous pytest test and waits for its completion.
        async with MCPAdapter(build_server(collection)) as adapter:
            # Discovery returns executable LangChain tool adapters and their
            # schemas. It does not run a search; ainvoke below executes one.
            tools = await adapter.list_tools()
            assert [tool.name for tool in tools] == ["search_wikipedia"]
            schema = tools[0].args_schema
            assert set(schema["properties"]) == {"query"}
            result = await tools[0].ainvoke({"query": "raven"})
            assert "raven-" in str(result)
            for query in ["", "x" * 501]:
                result = await tools[0].ainvoke({"query": query})
                assert '"passages": []' in str(result)
            model = MeteredDemoModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_wikipedia",
                                "args": {"query": "raven"},
                                "id": "lookup",
                            }
                        ],
                    ),
                    AIMessage(content="Ravens adapt to urban environments."),
                ]
            )
            agent = create_deep_agent(model=model, tools=tools, backend=StateBackend())
            final = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Where do ravens live?"}]}
            )
            # Deep Agent ainvoke returns graph state. Its ToolMessage is the
            # executed MCP search result; AIMessage.tool_calls was only the
            # preceding proposal to search, and the last message is the answer.
            observation = next(m for m in final["messages"] if m.type == "tool")
            assert observation.status == "success"
            # MCP evidence may be wrapped in text content blocks by the adapter.
            # Unwrap text before parsing the JSON search payload.
            content = observation.content
            if isinstance(content, list):
                content = next(
                    block["text"] for block in content if block["type"] == "text"
                )
            passages = json.loads(content)["passages"]
            assert len(passages) == 4
            assert all(p["source"] == "fixture" for p in passages)
            assert (
                final["messages"][-1].content == "Ravens adapt to urban environments."
            )

    asyncio.run(exercise())


# A child process proves the stdio transport contract separately from
# the in-process adapter, catching command and serialization regressions.
def test_stdio_discovery_and_call(tmp_path):
    script = tmp_path / "fixture_server.py"
    script.write_text('''"""Serve deterministic retrieval over real stdio for the transport test."""
from lg_report.mcp_servers.wikipedia import build_server
class Collection:
    def count(self):
        return 1
    def query(self, **kwargs):
        return {"ids": [["fixture-1"]], "documents": [["Raven evidence"]],
                "metadatas": [[{"source": "fixture"}]], "distances": [[0.0]]}
build_server(Collection()).run(transport="stdio")
''')

    async def exercise():
        # Verify discovery and a tool call over a real child-process connection.
        # The temporary script supplies deterministic evidence; leaving MCPAdapter
        # closes this transport lifetime before asyncio.run returns to the test.
        transport = StdioTransport(command=sys.executable, args=[str(script)])
        async with MCPAdapter(transport) as adapter:
            (tool,) = await adapter.list_tools()
            result = await tool.ainvoke({"query": "raven"})
            assert "fixture-1" in str(result)

    asyncio.run(exercise())


# The command-line server must fail clearly before serving an absent
# index, rather than silently producing an empty or misleading service.
def test_server_refuses_missing_index(tmp_path):
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lg_report.mcp_servers.wikipedia",
            "--directory",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode != 0
    assert "Build the index first" in result.stderr
    assert not result.stdout
