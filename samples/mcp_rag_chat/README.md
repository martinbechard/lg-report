<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Deep Agents with Wikipedia MCP

Run a Deep Agent that discovers `semantic_search_wikipedia` from a local FastMCP server,
searches the completed WikiText-103 Chroma index over stdio, and consumes the
returned passages in a second model request. The standard recorder saves
`run.json`, `spans.jsonl`, `prices.json`, and `report.html`.

## Run

From the repository root:

```sh
uv sync --locked
# Only if the existing RAG index has not been built:
uv run python -m samples.rag_chat.ingest
uv run python -m agent_runtime --sample mcp_rag_chat
```

No separately running server is needed. The workflow launches and closes the MCP
subprocess for each turn, using the same Python environment. The printed HTML
path defaults to `reports/mcp_rag_chat/report.html`, replacing the previous default run.

The default uses a scripted model with **real MCP transport and vector retrieval**.
It makes two simulated model calls and one real `semantic_search_wikipedia` call. The final
scripted message acknowledges retrieval; inspect the tool observation for actual
passages and IDs. It does not claim to generate or evaluate a factual answer.
No model API key is needed. The completed index and cached embedding model are
required; see [RAG setup](../rag_chat/README.md).

The report shows the retrieved evidence entering the second model request.
Simulated token counts are teaching estimates, not provider tokenizer counts.
Local embedding CPU and MCP transport have no separate LLM token charge.

## Ask your own questions

```sh
cp samples/mcp_rag_chat/.env.example samples/mcp_rag_chat/.env
# Configure the provider, model, and API key in that file.
uv run python -m agent_runtime --sample mcp_rag_chat --client console --live
```

Use `/quit` to finish and export the report. `/attach PATH` supplies UTF-8 text to
the conversation; it does not add documents to the index. Live mode uses actual
provider usage and asks the agent to cite passage IDs and abstain when evidence
is insufficient. The corpus is a historical subset, not current Wikipedia.

Standard options include `--out`, `--prices`, `--fx-file`, `--env-file`, and
`--metadata-only`. Model-price refreshes can use the network even with a scripted
model; supply `--prices models.json` to avoid them. FX reads the saved shared
`exchange-rate.json`. Credentials remain ignored by Git; stable sample reports
are included, and reruns replace their previous generated output.

## Code path

`sample.json` → `workflows/mcp_rag_chat.py` → `agents/wikipedia_mcp_agent.py`
→ `MCPAdapter` → stdio → `mcp_servers/wikipedia.py` → `tools/semantic_search_wikipedia.py`.

The synchronous workflow bridge preserves the recorder's callbacks and passes the
complete conversation history into async invocation. Each turn starts a fresh
agent and server connection; only conversation messages persist across turns.
It does not provide checkpoint/resume or persistent agent filesystem state.
Async applications can use `open_agent` directly for a longer-lived connection.
See [MCP server details](../../src/agent_runtime/mcp_servers/README.md).

## Verify

```sh
uv run pytest -q tests/test_mcp_rag_sample.py tests/test_wikipedia_mcp.py
```

Fixture tests verify callback propagation, reporting, and failure cleanup without
paid model calls. The standalone command above exercises the actual local index.
