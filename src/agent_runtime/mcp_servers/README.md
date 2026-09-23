<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Local MCP servers

`wikipedia.py` exposes the existing `tools/semantic_search_wikipedia.py` function through
FastMCP 4. The server opens a completed Chroma index once and publishes one
read-only tool, `semantic_search_wikipedia(query: str)`. The tool retains its 500-character
query limit and returns JSON with at most four passages, stable IDs, source
metadata, and distances. Empty or oversized queries return an explanatory note.
The corpus is a historical WikiText-103 subset, not live Wikipedia.

## Setup and launch

From the repository root:

```sh
uv sync --locked
# Only needed if the index has not already been built:
uv run python -m samples.rag_chat.ingest
# Usually launched by the MCP client, not manually:
uv run python -m agent_runtime.mcp_servers.wikipedia
```

The server defaults to the existing RAG sample index. Operators may select a
completed alternative with `--directory /absolute/path/to/index`. A missing or
partial index fails at startup. The model cannot select paths. Stdio requires
no port, HTTP service, or credentials; stdout is reserved for MCP messages.
See [RAG setup](../../../samples/rag_chat/README.md) for dataset and embedding details.

An MCP host can launch it using this configuration (adjust the checkout path):

```json
{
  "mcpServers": {
    "wikipedia": {
      "command": "uv",
      "args": ["run", "--project", "/Users/martinbechard/dev/lg-report", "--locked", "python", "-m", "agent_runtime.mcp_servers.wikipedia"]
    }
  }
}
```

## Deep Agents integration

Use the supplied async context with an already configured chat model:

```python
from agent_runtime.agents.wikipedia_mcp_agent import open_agent


async def answer(model, question):
    async with open_agent(model) as agent:
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
```

`open_agent` launches the server with the active Python interpreter, discovers its
tools, and supplies them to `create_deep_agent`. It keeps the connection open
until the context exits. Use `ainvoke`/`astream` for the asynchronous MCP tools.
The optional `directory=Path(...)` parameter is application configuration.
Existing synchronous RAG samples continue using their direct local tool.

Deep Agents needs no special server or custom tool wrapper. Current LangChain
provides `MCPAdapter` in `langchain.mcp` through the `langchain[mcp]` extra. It
converts MCP definitions into ordinary LangChain tools for the agent's `tools`
parameter. This API is currently beta; the project lockfile records tested
versions. Older examples use the separate `langchain-mcp-adapters` package and
`MultiServerMCPClient`; that is not the API used here.

To connect directly to another FastMCP server:

```python
from deepagents import create_deep_agent
from langchain.mcp import MCPAdapter


async def answer_from_server(model, question, server_url):
    async with MCPAdapter(server_url) as adapter:
        agent = create_deep_agent(model=model, tools=await adapter.list_tools())
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
```

The local server performs no chat-model requests. Passage text becomes model
input when the agent consumes it; MCP adds transport, not separate LLM billing.
This helper does not itself create an HTML report or change `run.json` accounting.

## Verification

```sh
uv run pytest -q tests/test_wikipedia_mcp.py tests/test_rag.py
```

Tests cover real MCP discovery, a tiny real Chroma collection, a scripted Deep
Agent tool loop, a stdio subprocess, input limits, and missing-index startup.
They need no API keys or downloads and do not assess live model answer quality.

References checked September 18, 2026:
[LangChain MCP](https://docs.langchain.com/oss/python/langchain/mcp),
[Deep Agents integration and migration](https://www.langchain.com/blog/mcp-in-langchain-stateless-protocol-elicitation-and-more),
[FastMCP](https://gofastmcp.com/).

For a runnable application with HTML and token reporting, see the
[MCP RAG sample](../../../samples/mcp_rag_chat/README.md).
