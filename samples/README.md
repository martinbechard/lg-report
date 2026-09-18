<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Training applications

Each directory contains an independently runnable DeepAgents/LangGraph application.
Start with each sample README. `app.py` composes a shared workflow with a client
and recorder. `test_case.py` owns scenario prompts and model fixtures; Langfuse
variants reuse the corresponding local scenario. No agent knows the test prompts.
See [composition diagrams](../docs/chat-composition.md).

| Application | What it teaches | Offline execution |
| --- | --- | --- |
| [Simple chat](simple_chat/README.md) | Conversation history across two user turns | 2 model calls, no tool calls |
| [Tool chat](tool_chat/README.md) | Model/tool/model routing and observations | 4 model calls, 2 tool calls |
| [Simple chat with Langfuse](simple_chat_langfuse/README.md) | Official callback, trace hierarchy, and conversation grouping | 2 model calls; requires Langfuse |
| [Parent and subagent](subagent_chat/README.md) | Isolated delegation and combined model costs | 4 model calls, 2 tool executions |
| [Parent and subagent with Langfuse](subagent_chat_langfuse/README.md) | Nested tracing of the same delegation | 4 model calls; requires Langfuse |
| [RAG with Chroma](rag_chat/README.md) | Retrieve from millions of indexed tokens | 2 model calls, 1 real vector search; ingest first |
| [MCP RAG](mcp_rag_chat/README.md) | Deep Agents tool discovery and Wikipedia retrieval over stdio MCP | 2 model calls, 1 MCP search; ingest first |
| [Review loop](review_loop/README.md) | Explicit graph, judge feedback, bounded revisions | 4 model calls, 2 drafts, 1 turn |
| [Expert dispatcher](expert_dispatch/README.md) | Model-selected delegation to movies, sports, or history | 12 model calls, 3 task executions + 3 retrievals, 3 turns |
| [Thinking agent](thinking_agent/README.md) | Evidence, reasoning cost, and verification | 7 model calls, 6 tool calls |

Run from the repository root after `uv sync`:

```bash
uv run python -m samples.simple_chat.app
uv run python -m samples.tool_chat.app
uv run python -m samples.thinking_agent.app
```

The three commands above print a new report path. Reports default to
`reports/<application>/<timestamp>-<id>/`, so runs do not overwrite one another.
`--out` selects a new directory. `--help` lists the supported options.

The [Langfuse variant](simple_chat_langfuse/README.md) has its own configuration
and command. It sends traces to Langfuse and prints a trace URL instead of
creating local reports. Its simulated model still requires a Langfuse project.

## Read the code in this order

1. **`app.py`** selects models, a workflow, a user client, and recording.
2. **`src/lg_report/workflows/`** composes participants, including single-agent workflows.
3. **`src/lg_report/agents/`** contains one named role per file, its instructions and graph/specification.
4. **`src/lg_report/tools/`** contains callable evidence tools, independent of clients.
5. **`test_case.py`** holds user prompts and scripted decisions; it never executes tools itself.
6. **`src/lg_report/platform/`** provides the common client loop and tracing lifecycle.
7. **`src/lg_report/report/`** captures, normalizes, prices, and exports local traces.

All applications use `--client static` by default. Use `--client console --live`
for interactive prompts and UTF-8 text attachments. `/attach PATH` queues a file,
`/send` submits attachments alone, and `/quit` ends the session. Console mode
requires a configured provider; static mode can use either simulated or live models.
There is one client loop, with no older prompt-list adapter or launch aliases.

## Model selection and configuration

Default runs use an offline model with a stateful context simulator. No LLM API
key is required. To use a provider, copy the application's `.env.example` to its
own `.env`, set an OpenAI or Anthropic key, and run that application with `--live`.
Shell environment variables take precedence. `--env-file` selects another file.
Provider behavior, reasoning visibility, cache hits, and call counts can differ
from the offline sequence; the report uses actual reported provider usage.

The daily EUR lookup uses the free Frankfurter service and a shared local cache.
Supply `--fx-file /path/to/usd-eur.json` to avoid a network lookup. Its format is
`{"rate":"0.871","date":"2026-09-17","base":"USD","quote":"EUR"}`; use your
chosen rate and its actual reference date. The report highlights stale dates.


## What the simulation means

Scripted responses still travel through a real graph and real local tools. The
simulator counts canonical message/tool-definition JSON words and punctuation,
not provider tokenizer tokens. It retains the visible conversation, including
each response, and assumes those tokens can be read from cache on the next call.
There is no wall-clock cache expiration in the simulator; the app records a `5m`
cache policy. A displayed simulated cache write is a context movement, distinct
from provider-reported billable cache creation. Hidden reasoning counts are
charged as output but are not appended as visible conversation history.

These distinctions are deliberate teaching assumptions, not promises about a
provider's cache implementation. Live mode does not synthesize cache hits or
reasoning tokens. Content is captured by default; `--metadata-only` omits message
and tool payloads. Generated reports and `.env` files are excluded from Git.

The current tool sample is a local lookup, not a vector RAG pipeline. The [Chroma RAG sample](rag_chat/README.md) adds semantic retrieval over a large
corpus. A file-editing application with human approval remains a later lesson; these examples do not claim to implement them.

The delegation lesson has matching [local](subagent_chat/README.md) and
[Langfuse](subagent_chat_langfuse/README.md) applications. Both use the same graph
and model fixtures; each README provides its launch command and configuration.
