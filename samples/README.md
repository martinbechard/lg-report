<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Training applications

Each directory describes a DeepAgents/LangGraph sample discovered by the shared
`SampleCatalog`. Start with its README. `sample.json` declares the sample identity,
workflow entry point, script module, and default options. `scripted_run.py` owns scenario prompts and model fixtures; Langfuse
variants reuse the corresponding local scenario. No agent knows the test prompts.
See [composition diagrams](../docs/chat-composition.md).

| Application | What it teaches | Offline execution |
| --- | --- | --- |
| [Simple chat](simple_chat/README.md) | Conversation history across two user turns | 2 model calls, no tool calls |
| [Shell script](shell_script/README.md) | Native execute tool and SandboxBackendProtocol | 2 model calls, 1 real shell script |
| [Tool chat](tool_chat/README.md) | Model/tool/model routing and observations | 4 model calls, 2 tool calls |
| [Simple chat with Langfuse](simple_chat_langfuse/README.md) | Official callback, trace hierarchy, and conversation grouping | 2 model calls; requires Langfuse |
| [Parent and subagent](subagent_chat/README.md) | Isolated delegation and combined model costs | 4 model calls, 2 tool executions |
| [Context budgets](context_budget/README.md) | Shared peer history, summarization, and an independently budgeted isolated subagent | 17 model calls including 5 summaries, 6 tool executions, 2 turns |
| [Parent and subagent with Langfuse](subagent_chat_langfuse/README.md) | Nested tracing of the same delegation | 4 model calls; requires Langfuse |
| [RAG with Chroma](rag_chat/README.md) | Retrieve from millions of indexed tokens | 2 model calls, 1 real vector search; ingest first |
| [MCP RAG](mcp_rag_chat/README.md) | Deep Agents tool discovery and Wikipedia retrieval over stdio MCP | 2 model calls, 1 MCP search; ingest first |
| [Review loop](review_loop/README.md) | Explicit graph, judge feedback, bounded revisions | 4 model calls, 2 drafts, 1 turn |
| [Expert dispatcher](expert_dispatch/README.md) | Model-selected delegation to movies, sports, or history | 12 model calls, 3 task executions + 3 retrievals, 3 turns |
| [File approval](file_approval/README.md) | Automatically intercept restricted tools chosen by an agent | 6 scripted model calls, 3 reads, 2 writes when approved |
| [Quote request](quote_request/README.md) | LLM spots conflicting requirements and asks the human | Live model + console; explicit offline simulation |
| [Thinking agent](thinking_agent/README.md) | Evidence, reasoning cost, and verification | 7 model calls, 6 tool calls |

Run from the repository root after `uv sync`:

```bash
uv run python -m agent_runtime --sample simple_chat
uv run python -m agent_runtime --sample tool_chat
uv run python -m agent_runtime --sample thinking_agent
```

Each command writes **`reports/<sample>/report.html`** and prints its absolute
path. Open it directly; rendering is automatic. Supporting evidence lives beside
it. Rerunning replaces that sample's previous generated output. `--out PATH`
selects another reusable directory. No dates or run IDs are added to folder names.
See [file roles and report commands](../README.md#where-files-go) for the
reference inputs, generated caches, and execution outputs.

To run **all local samples and export HTML plus Excel** in one batch with real models:

```sh
uv run scripts/run_samples.py
open reports/index.html
```

Real calls use `.env.local` (or `--env-file PATH`); shell settings win. The quote
sample asks questions in the terminal. For an unattended offline run, use
`uv run scripts/run_samples.py --simulated`.

The live batch refreshes the shared `exchange-rate.json` once before running
samples. Individual samples only read the saved file. HTML, Excel, and supporting
evidence are included in Git so you can inspect [saved results](../reports/index.html)
without running a model.

This refreshes `reports/<sample>/` for all local samples and excludes
Langfuse. `--out` selects a different batch directory. See the
[batch runner setup and output details](../README.md#run-all-local-samples-and-export-excel).

The [Langfuse variant](simple_chat_langfuse/README.md) has its own configuration
and command. It sends traces to Langfuse and prints a trace URL alongside
the local report bundle. Its simulated model still requires a Langfuse project.

## Shared launcher and discovery

```sh
uv run python -m agent_runtime --list
uv run python -m agent_runtime --sample simple_chat --client console --live
uv run python -m agent_runtime --sample subagent_chat --client angular
```

The console accepts `/samples`, `/sample tool_chat`, and `/new`. Selection closes
the previous conversation before creating a new graph, model, and checkpoint.
`/quit` exits. Approval/clarification answers are passed to the workflow unchanged.

To register another lesson, add a folder with `sample.json`:

```json
{
  "samples": [{
    "id": "my_lesson",
    "name": "My lesson",
    "description": "A longer explanation of what this lesson demonstrates.",
    "workflow": "agent_runtime.workflows.my_lesson:build_workflow",
    "scripted_run": "samples.my_lesson.scripted_run",
    "options": {}
  }]
}
```

Provide that workflow and script module, then restart the launcher. Discovery
reads JSON without importing workflows or creating models. IDs must be unique.
One folder can declare multiple variants, as `claims_context` does. Optional
`tracing: "langfuse"`, `interaction: "approval"` or `"clarification"`, and
`mcp_tools` describe harness behavior. `default_client` defaults to `static`;
file approval declares `console` so its default still asks a human before editing. No shared registration code changes.

Workflows obtain models through `build_model`; agents may request their own with
a named caller. A script can provide `CONVERSATION` entries (`client`, `ai`, named
agents, and documented tool observations), as `subagent_chat` does. Existing
specialized scripts expose `USER_PROMPTS` and `build_models(options)` so real tool
results and context-accounting simulations retain their behavior. Python
`ScriptPrompter` and browser `WebScriptPrompter` each own their prompt position;
the catalog only supplies data. Neither prompter supplies model responses.

## Read the code in this order

1. **`sample.json`** declares the ID, short name, explanation, workflow, and script.
   The shared `python -m agent_runtime` launcher selects clients and recording.
2. **`src/agent_runtime/workflows/`** composes participants, including single-agent workflows.
3. **`src/agent_runtime/agents/`** contains one named role per file, its instructions and graph/specification.
4. **`src/agent_runtime/tools/`** contains callable evidence tools, independent of clients.
5. **`scripted_run.py`** holds user prompts and scripted decisions; it never executes tools itself.
6. **`src/agent_runtime/harness/`** provides the common client loop and tracing lifecycle.
7. **`src/reporting/`** captures, normalizes, prices, and exports local traces.

Model-based applications use `--client static` by default. The file-approval sample defaults to console approval with scripted model decisions; `--live` uses the configured model with the same tool gate.
Quote-request clarification uses `--live --client console` for model judgment; its default is an explicitly scripted offline demonstration.
For chat applications, use `--client console --live`
for interactive prompts and UTF-8 text attachments. `/attach PATH` queues a file,
`/send` submits attachments alone, and `/quit` ends the session. Console mode
requires a configured provider; chat static mode can use either simulated or live models.
Quote clarification uses the shared console for interruption answers and requires a human client in live mode.
All samples use LangGraphAgent. Console and scripted clients consume AG-UI events directly; Angular uses AG-UI over HTTP/SSE. Approval and clarification are normal resumable interactions.

## Model selection and configuration

Default model-based runs use an offline model with a stateful context simulator. No LLM API
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

The current tool sample echoes the supplied text with a `Your input was: ` prefix. The [Chroma RAG sample](rag_chat/README.md) adds semantic retrieval over a large
corpus. The [file approval sample](file_approval/README.md) adds per-modification human approval,
and the [quote request sample](quote_request/README.md) adds model-directed clarification of multiple conflicting requirements and cancellation.

The delegation lesson has matching [local](subagent_chat/README.md) and
[Langfuse](subagent_chat_langfuse/README.md) applications. Both use the same graph
and model fixtures; each README provides its launch command and configuration.
