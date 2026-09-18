# LG Report

## Project structure

```text
src/lg_report/
  platform/   # Shared clients, sessions, model configuration and simulation
  workflows/  # Compose agents; one workflow even for a single agent
  agents/     # One agent per file, named after the agent; no test prompts
  tools/      # Agent-callable application tools
  report/     # Trace capture, normalized data, prices, HTML and Excel
samples/
  simple_chat/  # app.py wiring, test_case.py fixtures, README and configuration
```

All samples use the same client/conversation architecture. Agent definitions,
workflow composition, and callable tools live in their shared packages. Sample
directories contain entry points, test cases, READMEs, and configuration examples.
Langfuse variants reuse the corresponding workflow and test case. There are no
old import aliases or separate legacy conversation loops.


A single-user Python library that captures LangGraph / DeepAgents execution and produces local HTML and Excel reports of conversations, spans, token usage, and estimated model costs.

## Training applications

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install once:

```sh
uv sync --locked
```

Each sample wires a workflow to its client and tracing backend; its test case owns prompts and scripted responses. Run commands from the repository root:

| Application | Purpose | Command |
| --- | --- | --- |
| [Simple chat](samples/simple_chat/README.md) | Conversation history across two user turns | `uv run python -m samples.simple_chat.app` |
| [Tool chat](samples/tool_chat/README.md) | Model request → tool → model request, across two turns | `uv run python -m samples.tool_chat.app` |
| [RAG with Chroma](samples/rag_chat/README.md) | Large-corpus ingestion and cited retrieval | `uv run python -m samples.rag_chat.app` (restores bundled index on first use) |
| [Review loop](samples/review_loop/README.md) | High-level draft → judge feedback → detailed revision | `uv run python -m samples.review_loop.app` |
| [Expert dispatcher](samples/expert_dispatch/README.md) | Select movies, sports, or history expert for each question | `uv run python -m samples.expert_dispatch.app` |
| [Thinking agent](samples/thinking_agent/README.md) | Reasoning costs before and after several tool calls | `uv run python -m samples.thinking_agent.app` |

The [Langfuse variant of simple chat](samples/simple_chat_langfuse/README.md)
uses the official Langfuse callback and displays traces in your configured
Langfuse project. It has separate setup instructions and does not generate local reports.

The parent/subagent lesson also has matching [local-report](samples/subagent_chat/README.md)
and [Langfuse](samples/subagent_chat_langfuse/README.md) applications.

Start with the [sample catalog](samples/README.md). Default execution uses a real graph with a simulated model and makes no model-provider request. To run live, copy that sample's `.env.example` to its own `.env`, configure the selected provider, and add `--live` to its command. Existing environment variables take precedence. There is no automatic switch between live and simulated execution.

Each local-report application prints its HTML path. `--out` selects a new output directory; existing directories are rejected. Every run writes:

| File | Purpose |
| --- | --- |
| `spans.jsonl` | Raw local OpenTelemetry SDK spans |
| `run.json` | Validated, provider-independent report data |
| `prices.json` | Exact pricing and exchange-rate snapshot |
| `report.html` | Self-contained report with chart, conversation, execution tree, and references |

The samples capture message and tool content by default. `--metadata-only` omits those payloads. Exception messages are omitted; exception types identify failures. `.env` files and generated reports are ignored by Git. Reporting uses a local exporter and disables inherited LangSmith tracing for the invocation; live prompts still go to the selected model provider.

## Report commands

The reporting CLI processes saved data. Applications launch through their own modules.

```sh
uv run lg-report render reports/my-run/run.json --out reports/my-run/report.html
uv run lg-report normalize reports/my-run/spans.jsonl --title "Imported run" --out reports/my-run/rebuilt.json
```

`render` uses the adjacent `prices.json`; `--prices models.json` explicitly selects another table. It obtains the current daily exchange rate. `normalize` accepts this application's OTel SDK JSONL format, not arbitrary OTLP collector JSON. Use its `--demo` flag when importing a simulated trace. Both commands replace their specified output file. `run.json` is the portable report boundary.

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

Every cost identifies EUR or USD. The reference section records price verification dates and exchange-rate provenance. Older dates are highlighted at report generation time; regenerating reassesses freshness.

## Daily EUR conversion

The applications and renderer use [Frankfurter's free API](https://frankfurter.dev/) for the latest published ECB USD/EUR rate. Successful lookups are saved in `.cache/lg-report/fx/YYYY-MM-DD.json`, keyed by the machine's local lookup date. Later starts that day reuse the file. Weekends and holidays can have an older reference date.

Use `--fx-file exchange-rate.json` on a sample command to supply a rate without a lookup. On the reporting CLI, place `--fx-file` before `render`. `LG_FX_FILE` and `LG_FX_CACHE` configure the same behavior through the environment.

```json
{"base":"USD","quote":"EUR","rate":"0.871","date":"2026-09-17","source":"User-supplied reference rate"}
```

The rate means EUR per USD. This is a dated example, not a current rate. The report embeds the exact rate, source, reference date, and optional fetch timestamp. If conversion is unavailable, EUR remains unknown and USD is retained. A corrupt cache is reported rather than silently replaced.

## Instrument another application

```python
from pathlib import Path
from lg_report.report.pricing import load_prices
from lg_report.report.recording import record_run

# Wrap the compiled graph at its execution boundary so nested callbacks
# preserve the graph hierarchy in the report.
record_run(
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

The provider and model identify the configured model when callback metadata does not supply them. Multi-model graphs should emit `ls_provider` and `ls_model_name`. Direct `record_run` calls default to metadata-only capture. `Conversation(workflow, StaticClient([Request(prompt1), Request(prompt2)]))` preserves history and supplies turn metadata when recording multiple user turns.

Use `run_name`, tool docstrings, and `report_description` / `report_purpose` metadata to explain operations. Keep descriptions specific to the runnable's purpose: metadata can be inherited by children. See the sample graph definitions for complete examples. Unlisted metadata is not exported.

The current runner supports synchronous graph invocation. Streaming, provider-internal retry accounting, external embedding charges, and durable interrupt/resume workflows remain outside its scope. The Chroma RAG sample provides local vector retrieval. File editing with human approval and CSV export remain future samples/features.

## Excel export

The workbook has Turns, Execution tree, and Reference data sheets. All freeze row 1 and column A. Turns follows the HTML request/context/response/tool sequence. Tokens, EUR, and USD occupy separate columns. Reference data B2 sets executions (default 100,000); projected costs round to zero decimal places only after multiplication. Per-execution values retain precision.

```sh
uv run python -m lg_report.report.excel_data reports/my-run/run.json --out .cache/excel/data.json
node src/lg_report/report/export_excel.mjs .cache/excel/data.json outputs/lg-report-excel/report.xlsx
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

[Regenerated sample reports](reports/examples/README.md) include HTML and their
accounting evidence. Other local runs and Langfuse databases/reports are ignored.
Compressed RAG assets and extraction details are in [data/rag](data/rag/README.md).
