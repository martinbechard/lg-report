# LG Report

Run every sample through the shared launcher: `uv run python -m agent_runtime --list`
lists the discovered IDs. Use `--sample subagent_chat` for a scripted run, or
`--sample simple_chat --client console --live` for interactive chat. The console
supports `/samples`, `/sample ID`, and `/new`. Sample folders register themselves
through `sample.json`; see [adding a sample](samples/README.md#shared-launcher-and-discovery).

## Project structure

```text
src/
  agent_runtime/
    harness/     # Clients, sessions, model configuration, simulation and capture
    workflows/   # Compose agents; one workflow even for a single agent
    agents/      # One agent per file; no test prompts
    tools/       # Agent-callable application tools
    mcp_servers/ # Tool servers exposed over MCP
    web/         # HTTP endpoints; catalog lives in harness/
  reporting/     # Normalized data, prices, HTML, Excel and reporting CLI
samples/
  simple_chat/   # sample.json metadata, scripted_run.py fixtures, README and configuration
```

Runtime imports use `agent_runtime`; report imports use `reporting`. The installed
`lg-report` command and `uv run python -m reporting` invoke the same reporting CLI.

All samples use the same client/conversation architecture. Agent definitions,
workflow composition, and callable tools live in their shared packages. Sample
directories contain entry points, test cases, READMEs, and configuration examples.
Langfuse variants reuse the corresponding workflow and test case.

See [context management](docs/chat-composition.md#context-management) for shared
peer histories, isolated or forked subagents, and summarization `trigger` and
`keep` settings. The [context-budget sample](samples/context_budget/README.md)
demonstrates independent workflow and subagent budgets.


A single-user Python library that captures LangGraph / DeepAgents execution and produces local HTML and Excel reports of conversations, spans, token usage, and estimated model costs.

## Training applications

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install once:

```sh
uv sync --locked
```

Each sample wires a workflow to its client and tracing backend; its test case owns prompts and scripted responses. Run commands from the repository root:

| Application | Purpose | Command |
| --- | --- | --- |
| [Simple chat](samples/simple_chat/README.md) | Conversation history across two user turns | `uv run python -m agent_runtime --sample simple_chat` |
| [Tool chat](samples/tool_chat/README.md) | Model request → tool → model request, across two turns | `uv run python -m agent_runtime --sample tool_chat` |
| [RAG with Chroma](samples/rag_chat/README.md) | Large-corpus ingestion and cited retrieval | `uv run python -m agent_runtime --sample rag_chat` (restores bundled index on first use) |
| [Review loop](samples/review_loop/README.md) | High-level draft → judge feedback → detailed revision | `uv run python -m agent_runtime --sample review_loop` |
| [Expert dispatcher](samples/expert_dispatch/README.md) | Select movies, sports, or history expert for each question | `uv run python -m agent_runtime --sample expert_dispatch` |
| [File approval](samples/file_approval/README.md) | Agent-invoked tools with automatic human approval for restricted writes | See sample README for source/target arguments |
| [Quote request](samples/quote_request/README.md) | LLM-directed human clarification and cancellation | `uv run python -m agent_runtime --sample quote_request` |
| [Claim context](samples/claims_context/README.md) | Compare retained snapshots with purge/reload after edits | `uv run python -m agent_runtime --sample claims_context_managed` |
| [Thinking agent](samples/thinking_agent/README.md) | Reasoning costs before and after several tool calls | `uv run python -m agent_runtime --sample thinking_agent` |

The [Langfuse variant of simple chat](samples/simple_chat_langfuse/README.md)
uses the official Langfuse callback and displays traces in your configured
Langfuse project. It has separate setup instructions and adds Langfuse traces alongside local reports.

The parent/subagent lesson also has matching [local-report](samples/subagent_chat/README.md)
and [Langfuse](samples/subagent_chat_langfuse/README.md) applications.

Start with the [sample catalog](samples/README.md). Default execution uses a real graph with a simulated model and makes no model-provider request. To run live, copy that sample's `.env.example` to its own `.env`, configure the selected provider, and add `--live` to its command. Existing environment variables take precedence. There is no automatic switch between live and simulated execution.

### Run the same sample with Angular

The [Angular chat frontend](frontend/README.md) uses standard AG-UI SSE streaming and the
same sample factories and UTF-8 attachment formatting as the console.
Build it once with `npm --prefix frontend ci` and `npm --prefix frontend run build`.
Then select the frontend without changing the sample's workflow or agent:

```sh
uv run --extra chat python -m agent_runtime --sample simple_chat --live --client console
uv run --extra chat python -m agent_runtime --sample simple_chat --live --client angular
```

Open [the local chat](http://127.0.0.1:8000). Omit `--live` for a labelled scripted
demo. See the frontend README for supported samples, development, and parity tests.

### Run all local samples and export Excel

```sh
uv run scripts/run_samples.py
open reports/index.html
```

One command runs all local samples and generates **HTML and standalone Excel
workbooks**, with a clickable index. **Real models are the default**, using
`.env.local` for provider credentials and `LG_MODEL` (OpenAI defaults to
`gpt-5.6-luna`). Shell settings take precedence; `--env-file PATH` selects
another configuration. Langfuse samples are excluded. Requests and file approvals
are scripted; the live quote sample asks clarification questions in the terminal.

For an unattended offline batch, explicitly select simulated models:

```sh
uv run scripts/run_samples.py --simulated
```

Simulation needs no provider key and never substitutes for a failed live call.
RAG and MCP RAG still execute real local retrieval and restore the bundled index
on first use. The file-approval lesson writes only its own `edited-summary.txt`.

Outputs default to `reports/<sample>/report.html` and `report.xlsx`, relative
to your working directory. Each folder also contains the run evidence,
`excel-data.json`, workbook previews, and `run.log`. Repeating the command refreshes
these outputs. Use `--out reports/another-batch` to keep a separate batch.
Failures are listed in the index, remaining samples still run, and the command
exits nonzero if any sample or export fails.

The runner uses the checked-in pricing catalog without fetching prices. The live
batch first runs `scripts/update_exchange_rate.py` once to refresh the shared
`exchange-rate.json`. Failed refreshes preserve the saved rate. Individual samples
only read that file; missing or invalid FX leaves EUR unavailable without stopping
model execution. `--fx-file path/to/rate.json` and `--simulated` skip the refresh. `--prices path/to/models.json` overrides the catalog. The first RAG
execution may need the local embedding model downloaded if it is not cached.

Excel generation uses the existing `@oai/artifact-tool` exporter. The runner
finds the standard Codex desktop runtime automatically. On another installation,
set `LG_EXCEL_RUNTIME` to the directory containing its `node_modules`, with Node.js
on `PATH`. Dependencies are checked before samples start; nothing is installed
automatically. The generated HTML and `.xlsx` files can be opened independently
of Python, Node.js, Codex, and Langfuse.

### Where files go

Local samples write **`reports/<sample>/report.html`**. Open that file to see
results; no separate render command is needed. For example:

```sh
uv run python -m agent_runtime --sample simple_chat
open reports/simple_chat/report.html
```

Rerunning a sample replaces its previous generated report bundle, including stale
Excel exports. There are no dates or run IDs in folder names. Other samples keep
their own results. Claims-context modes use `claims_context_naive` and
`claims_context_managed`. Browser turns replace the latest report for that sample.
The command prints the absolute HTML path. `--out PATH` selects another directory
and replaces its generated reports too; unrelated files are preserved.
Relative paths are relative to the working directory.

| Run output | Purpose |
| --- | --- |
| `report.html` | The finished, self-contained report to open |
| `run.json` | Normalized execution data used to rebuild HTML or export Excel |
| `spans.jsonl` | Raw captured trace; input to `normalize` |
| `prices.json` | This run's saved pricing and exchange-rate snapshot |

These four files belong together. They describe one execution and are replaced
as a bundle on the next default sample run. Langfuse samples also send their
traces to Langfuse and print a trace URL; they create the same local bundle.

### Reference files versus run outputs

| Reference/configuration | Role |
| --- | --- |
| `models.json` | Shared pricing catalog used when starting runs; not a run result |
| `exchange-rate.json` | Saved USD/EUR reference; refreshed once by the live batch |
| `samples/<sample>/.env` | Provider configuration for that sample |
| `samples/<sample>/scripted_run.py` | Authored prompts and scripted model fixtures |
| `.cache/lg-report/prices/` and `.cache/lg-report/fx/` | Downloaded, reusable reference data; created dynamically, not execution reports |
| `data/rag/` and `.cache/lg-report/rag/` | Bundled reference corpus/index archives and the restored retrieval index |

The samples keep their catalog, default caches, and `.env` relative to the
repository/sample even when launched elsewhere. Output files go to your working
directory. `prices.json` is a per-run snapshot of reference data, whereas
`models.json` and the caches are reusable inputs.

The samples capture message and tool content by default. `--metadata-only` omits those payloads. Exception messages are omitted; exception types identify failures. Credentials and ad hoc runs are ignored by Git; stable sample reports and evidence are included. Reporting uses a local exporter and disables inherited LangSmith tracing for the invocation; live prompts still go to the selected model provider.

## Report commands

The sample has already produced HTML. Use these commands only to rebuild saved
outputs; neither command runs the agent:

```sh
uv run lg-report render reports/simple_chat/run.json
uv run lg-report normalize reports/simple_chat/spans.jsonl --demo
uv run lg-report render reports/simple_chat/run.json
```

For a retained run created with `--out reports/first-chat`:

```sh
uv run lg-report render reports/first-chat/run.json
uv run lg-report normalize reports/first-chat/spans.jsonl --demo --title "Imported run"
```

Input filenames are optional; the defaults are `run.json` and `spans.jsonl` in
the working directory. Output defaults to `report.html` or `run.json` **beside
the input file**. `--out PATH` overrides the output filename. These commands
replace their output file.

`render` reads the adjacent `prices.json`; `--prices models.json` explicitly
selects another table. It reads the shared saved exchange rate without a network lookup. `normalize`
accepts this application's OTel SDK JSONL format, not arbitrary OTLP collector
JSON. Use `--demo` only for simulated traces; omit it for live traces.
Normalization rebuilds execution data from spans; it does not rerun the agent
or preserve all original run-level metadata such as the title and final output.
`run.json` is the portable report boundary.

## Collection and accounting

```text
LangGraph / DeepAgents application
    → LangChain callbacks → local OpenTelemetry spans
    → validated Run → run.json
    + pricing snapshot → HTML / Excel
```

Only model spans contribute token costs. Graph and tool spans show execution structure without additional model charges. Requests include the conversation and tool results sent to the model; responses include tool requests. Turn totals count each model call once. Execution-tree parent rows show subtree totals and must not be summed with their children.

Real usage comes from provider-reported `AIMessage.usage_metadata`. Missing usage or prices remain unknown. The simulator maintains a growing context ledger, counts canonical message JSON and tool definitions, and assumes the completed conversation is cached for the next request. These are illustrative counts, not a provider tokenizer. Simulated reasoning text is a teaching fixture. Live reports show only reasoning content exposed by the provider.

`models.json` contains exact `provider:model` keys and USD rates per million tokens. Costs use `Decimal`. Cache categories partition input; reasoning partitions output and uses the output rate. The app's cache TTL is five minutes. Unknown models or unpriced categories produce an incomplete known subtotal. At sample startup, standard prices for GPT-5.5, GPT-5.6 Luna, GPT-5.6 Sol, Claude Sonnet 5, Claude Opus 4.8, and Claude Fable 5.1 are refreshed from official provider pages. Successful lookups and source text are saved under `.cache/lg-report/prices` by date and reused that day (`LG_PRICES_CACHE` overrides the directory). Demo rates explicitly follow GPT-5.6 Luna. `--prices` or `LG_PRICES` supplies an authoritative file and disables price fetching. Failed lookups retain the last verified prices and dates, with a warning in the report; unsupported models require a supplied file. Saved reports keep their exact `prices.json` snapshot, and `render` does not reprice historical runs automatically. Estimates exclude embedding calls, tool fees, infrastructure, taxes, and discounts.

Every cost identifies EUR or USD. The reference section records price verification dates and exchange-rate provenance. Saved dates provide provenance without marking every cost as an error. Exchange-rate publication can precede retrieval on weekends, holidays, or before the daily update; actual retrieval failures and missing data remain visible.

## Daily EUR conversion

The refresh script uses [Frankfurter's free API](https://frankfurter.dev/) for the latest published ECB USD/EUR rate and writes the shared, checked-in `exchange-rate.json`. The live batch invokes it once before starting samples. A lookup failure preserves the previous reference. Weekends and holidays can have an older publication date.

Individual samples and report rendering read the saved `exchange-rate.json` without network access. Run `uv run scripts/update_exchange_rate.py` to refresh it independently. `--fx-file PATH` or `LG_FX_FILE` selects another saved reference; on the reporting CLI, place `--fx-file` before `render`.

```json
{"base":"USD","quote":"EUR","rate":"0.871","date":"2026-09-17","source":"User-supplied reference rate"}
```

The rate means EUR per USD. This is a dated example, not a current rate. The report embeds the exact rate, source, reference date, and optional fetch timestamp. If conversion is unavailable, EUR remains unknown and USD is retained. A corrupt cache is reported rather than silently replaced.

## Instrument another application

```python
from pathlib import Path
from reporting.pricing import load_prices
from reporting.execute_runnable import execute_runnable

# Wrap the compiled graph at its execution boundary so nested callbacks
# preserve the graph hierarchy in the report.
execute_runnable(
    agent,
    {"messages": [("user", "Your question")]},
    Path("reports/custom-run"),
    load_prices(Path("models.json")),
    provider="openai",
    model="gpt-5.6-luna",
    title="My workflow",
    include_output=True,
)
```

The provider and model identify the configured model when callback metadata does not supply them. Multi-model graphs should emit `ls_provider` and `ls_model_name`. Direct `execute_runnable` calls default to metadata-only capture. `Conversation(workflow, ConsoleClient(prompter=ScriptPrompter([Request(prompt1), Request(prompt2)])))` preserves history and supplies turn metadata when recording multiple user turns.

Use `run_name`, tool docstrings, and `report_description` / `report_purpose` metadata to explain operations. Keep descriptions specific to the runnable's purpose: metadata can be inherited by children. See the sample graph definitions for complete examples. Unlisted metadata is not exported.

The current runner supports synchronous graph invocation. Streaming, provider-internal retry accounting, external embedding charges, and durable interrupt/resume workflows remain outside its scope. The Chroma RAG sample provides local vector retrieval. The file-approval and quote-request samples support in-process human interrupt/resume; both use agent decisions in live mode. CSV export remains a future feature.

## Excel export

The workbook has Turns, Execution tree, and Reference data sheets. All freeze row 1 and column A. Turns follows the HTML request/context/response/tool sequence. Tokens, EUR, and USD occupy separate columns. Reference data B2 sets executions (default 100,000); projected costs round to zero decimal places only after multiplication. Per-execution values retain precision.

```sh
uv run python -m reporting.excel_data run.json --out .cache/excel/data.json
node src/reporting/export_excel.mjs .cache/excel/data.json outputs/lg-report-excel/report.xlsx
```

The builder uses the desktop's bundled `@oai/artifact-tool`. Set `LG_EXCEL_RUNTIME` to the directory whose `node_modules` contains it. The exported workbook is independent of that runtime.

## Development checks

```sh
uv run pytest -q
uv run ruff check src samples tests
uv run ruff format --check src samples tests
uv build
```

Tests execute the documented sample commands with an offline model and a supplied exchange-rate file. Live provider API calls are not part of the suite.

See [Chat composition and tracing](docs/chat-composition.md) for Mermaid diagrams
and the shared client, agent, conversation, and recording design.

## License and checked-in examples

Copyright (c) 2026 Martin.Bechard@DevConsult.ca. Project software is licensed
under the [MIT License](LICENSE); third-party content retains its own terms
(see [NOTICE](NOTICE)).

[Saved sample results](reports/index.html) include HTML, Excel, and their accounting
evidence in Git, so learners can inspect them without running models. Ad hoc local
runs and Langfuse databases/reports remain ignored.
Compressed RAG assets and extraction details are in [data/rag](data/rag/README.md).

The [shell script sample](samples/shell_script/README.md) demonstrates DeepAgent’s native `execute` tool with a local `ShellBackend` implementing `SandboxBackendProtocol`.
