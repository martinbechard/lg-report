# Training applications

Each directory contains an independently runnable DeepAgents/LangGraph application.
Start with `app.py`: it owns the graph, system instructions, user prompts, and the
point where reporting is attached. None of these applications reads an existing
report to produce its trace.

| Application | What it teaches | Offline execution |
| --- | --- | --- |
| [Simple chat](simple_chat/README.md) | Conversation history across two user turns | 2 model calls, no tool calls |
| [Tool chat](tool_chat/README.md) | Model/tool/model routing and observations | 4 model calls, 2 tool calls |
| [Simple chat with Langfuse](simple_chat_langfuse/README.md) | Official callback, trace hierarchy, and conversation grouping | 2 model calls; requires Langfuse |
| [Parent and subagent](subagent_chat/README.md) | Isolated delegation and combined model costs | 4 model calls, 2 tool executions |
| [Parent and subagent with Langfuse](subagent_chat_langfuse/README.md) | Nested tracing of the same delegation | 4 model calls; requires Langfuse |
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

1. **`app.py`** — purpose, system/user prompts, graph construction, and entry point.
2. **`tools.py`**, where present — the actual functions the graph invokes.
3. **`simulation.py`** — the deterministic model responses used for teaching.
4. **`lg_report.sample_runtime`** — common configuration and trace attachment.
5. **`lg_report.runner.record_run`** — raw capture, normalization, pricing snapshot,
   and HTML rendering. The application does not implement reporting internals.

The shared runtime never chooses which application to run. Each application's
`main()` creates its own graph and passes it to the recorder. You can call
`build_agent(model)` and invoke that graph directly without generating a report.
The applications share the root Python environment and reporting library; they
do not require three duplicate dependency installations.

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

The current tool sample is a local lookup, not a vector RAG pipeline. A full RAG
application and a file-editing application with human approval remain later
lessons; these examples do not claim to implement them.

The delegation lesson has matching [local](subagent_chat/README.md) and
[Langfuse](subagent_chat_langfuse/README.md) applications. Both use the same graph
and model fixtures; each README provides its launch command and configuration.
