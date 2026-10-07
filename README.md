# LG Report

## Getting started

Run LangGraph samples and inspect local reports of conversations,
token usage, and estimated model costs. Run the commands below from the repository root.

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/), then install the project dependencies:

```sh
uv sync --locked
```

For a destination that uses **pip only**, use the [pip installation guide](PIP-INSTALL.md).
The repository includes the compiled browser interface. Node.js and npm are
needed only when changing and rebuilding the UI.

The optional RAG corpus and index are downloaded separately (about 342 MiB):

```sh
python scripts/download_rag.py
```

Run this before RAG samples or the complete sample batch. Other samples do not
need the RAG download. See [RAG setup](data/rag/README.md).
To avoid downloading old RAG blobs retained in Git history, use
`git clone --depth 1 https://github.com/martinbechard/lg-report.git`.

### Azure Foundry with GPT-4.1

Set the following in `.env.local`. Use your Azure resource's key and the exact
GPT-4.1 **deployment name** (which may differ from the model name):

```dotenv
LG_PROVIDER=openai
LG_MODEL=YOUR-GPT-4.1-DEPLOYMENT-NAME
OPENAI_API_KEY=YOUR-AZURE-RESOURCE-KEY
OPENAI_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
# Optional output limit; omitted by default for OpenAI.
# LG_MAX_TOKENS=32768
LG_EFFORT=
```

Then launch the UI with that file:

```sh
uv run --extra chat python -m agent_runtime --sample simple_chat --client angular --live --env-file .env.local
```

The endpoint is read directly from the file; no shell export is needed. Shell
variables still override matching file settings. Azure's OpenAI v1 endpoint uses
no `api-version` parameter. Leave reasoning effort empty for GPT-4.1. This setup
uses API-key authentication and the Responses API; use Azure-specific rates for
accurate cost estimates. See [Azure's Responses API documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses).

### GitHub Copilot local server

Use `LG_PROVIDER=copilot` with an explicit model code such as `LG_MODEL=gpt-5.4`.
Install the optional dependency with `uv sync --extra copilot`. The runtime starts
on demand on the first free port from 7001 and records the endpoint in local
configuration. This provider runs text request/response with all tools disabled.
See [Copilot installation, configuration, and model codes](docs/copilot-models.md).

### Run all samples as a batch

The batch runs all local samples and produces HTML reports, Excel workbooks,
and a clickable report index. Langfuse samples require separate tracing setup and are excluded.

Configure your provider in `.env.local` before a live batch. For example, set
`LG_PROVIDER=openai` and `OPENAI_API_KEY` there, or use `LG_PROVIDER=anthropic`
and `ANTHROPIC_API_KEY`. Shell environment variables take precedence.

```sh
uv run scripts/run_samples.py
```

Open [reports/index.html](reports/index.html) when the batch finishes.
Each sample has a `report.html` and `report.xlsx` under `reports/<sample>/`.
Live demos use a user agent to answer clarification questions automatically.

For an unattended batch with scripted model responses and no provider key:

```sh
uv run scripts/run_samples.py --simulated
```

Excel export uses XlsxWriter, installed by `uv sync --locked`. The batch needs no
Codex installation, Node.js, or `LG_EXCEL_RUNTIME` setting.
See [batch setup and output details](#run-all-local-samples-and-export-excel).

### Run one sample from the command line

Choose model behavior and whether the sample runs by itself:

| Command | Behavior |
| --- | --- |
| `--demo --static` | Unattended sample using fixed user and assistant text |
| `--demo --live` | Unattended conversation between real assistant and user agents |
| `--live --client console` | A human chats with the real assistant |
| `--live --client static` | Fixed user text sent to the real assistant |

`--demo` alone runs unattended and selects live agents when the assistant provider
is configured, otherwise fixed text. `--live` and `--static` are mutually exclusive.
Codex, Copilot, and API-key providers use the same adapters for either agent.

Use fixed text for a repeatable offline sample:

```sh
uv run python -m agent_runtime --sample simple_chat --demo --static
```

Open [reports/simple_chat/report.html](reports/simple_chat/report.html) after the run.
Individual sample commands produce HTML and supporting run data; the batch also exports Excel.
Rerunning replaces that sample's previous report bundle.

For human console chat, select the console client explicitly:

```sh
uv run python -m agent_runtime --sample simple_chat --client console --env-file .env.local
```

A configured key for the selected provider enables live mode automatically.
Without a key, the launcher uses scripted responses. `--live` explicitly requires
real execution; `--static` forces scripted responses even when a key is configured.
If you omit `--env-file`, the launcher reads the sample's own `.env` instead.

Use `/quit` to end console chat and write its report. `/samples` lists samples,
`/sample ID` switches samples, and `/new` starts a fresh conversation.
List available sample IDs from the command line:

```sh
uv run python -m agent_runtime --list
```

### Start the web server and Uvicorn API

The web launcher starts one Uvicorn server that serves both the Angular chat
and the FastAPI API. The prebuilt frontend is included in the checkout:

```sh
uv sync --locked --extra chat
```

Start the server with your provider configuration:

```sh
uv run --extra chat python -m agent_runtime --sample simple_chat --client angular --env-file .env.local
```

Add `--static` for scripted responses. The same provider-key detection applies as
in the console. Use `--port 8001` to select another port; stop the server with Ctrl-C.

- [Chat](http://127.0.0.1:8000/): select a sample and send messages.
- [API documentation](http://127.0.0.1:8000/docs): inspect the FastAPI endpoints.
- [Sample catalog API](http://127.0.0.1:8000/api/samples): inspect the available samples as JSON.

See the [frontend guide](frontend/README.md) for development and browser verification.

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
  simple_chat/   # sample.py metadata and fixtures, README and configuration
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

## Training applications

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
| [Claim context](samples/edit_with_reloaded_state/README.md) | Compare retained snapshots with purge/reload after edits | `uv run python -m agent_runtime --sample edit-with-reloaded-state` |
| [Thinking agent](samples/thinking_agent/README.md) | Reasoning costs before and after several tool calls | `uv run python -m agent_runtime --sample thinking_agent` |

The [Langfuse variant of simple chat](samples/simple_chat_langfuse/README.md)
uses the official Langfuse callback and displays traces in your configured
Langfuse project. It has separate setup instructions and adds Langfuse traces alongside local reports.

The parent/subagent lesson also has matching [local-report](samples/subagent_chat/README.md)
and [Langfuse](samples/subagent_chat_langfuse/README.md) applications.

Start with the [sample catalog](samples/README.md). The launcher defaults to live agents when the selected provider has an API key; otherwise it uses scripted responses and the sample’s default client (usually fixed prompts; file approval still asks a human). Use `--static` for fixed sample text even when a key is configured. `--demo` runs a sample unattended and can combine with either `--live` or `--static`. Configure the root `.env.example` as `.env.local` and pass `--env-file .env.local`; shell variables take precedence. The same root template includes Langfuse settings (see the [sample configuration guide](samples/README.md#model-selection-and-configuration)). `LG_PROVIDER` selects OpenAI (the default), Anthropic, Copilot, or Codex. For OpenAI and Anthropic, only that provider’s key enables automatic live mode; selecting Copilot or Codex enables live mode using its local login. `--live` explicitly requires real execution; provider errors never fall back to static text. `--static` and `--live` cannot be combined. For fixed prompts against a real model, use `--live --client static`.

### Run all local samples and export Excel

```sh
uv run scripts/run_samples.py
open reports/index.html
```

One command runs all local samples and generates **HTML and standalone Excel
workbooks**, with a clickable index. **Real models are the default**, using
`.env.local` for provider credentials and `LG_MODEL` (OpenAI defaults to
`gpt-5.6-luna`). Shell settings take precedence; `--env-file PATH` selects
another configuration. Langfuse samples are excluded. The user agent generates
requests and clarification answers; file approvals follow the explicit batch test policy.

To restrict model selection, set `LG_AVAILABLE_MODELS=model-id,other-model-id`
in `.env.local` (use exact deployment names for Azure). Unset or blank means
unrestricted. A requested model outside the list logs a warning once per model
per process, explaining the fallback to `LG_MODEL`. The default must also be in
the list; if it is missing or unavailable, selection raises an error. Both the
adapter and report identity use the resolved model. This checks the configured
list; provider API failures still propagate without automatic fallback.

For an unattended offline batch, explicitly select simulated models:

```sh
uv run scripts/run_samples.py --simulated
```

Simulation needs no provider key and never substitutes for a failed live call.
RAG and MCP RAG still execute real local retrieval and restore the bundled index
on first use. The file-approval lesson writes only its own `edited-summary.txt`.

Outputs default to `reports/<sample>/report.html` and `report.xlsx`, relative
to your working directory. Each folder also contains the run evidence,
`excel-data.json` and `run.log`. Repeating the command refreshes
these outputs. Use `--out reports/another-batch` to keep a separate batch.
Failures are listed in the index, remaining samples still run, and the command
exits nonzero if any sample or export fails.

The runner uses the checked-in pricing catalog without fetching prices. The live
batch first runs `scripts/update_exchange_rate.py` once to refresh the shared
`exchange-rate.json`. Failed refreshes preserve the saved rate. Individual samples
only read that file; missing or invalid FX leaves EUR unavailable without stopping
model execution. `--fx-file path/to/rate.json` and `--simulated` skip the refresh. `--prices path/to/models.json` overrides the catalog. The first RAG
execution may need the local embedding model downloaded if it is not cached.

Excel generation uses the project's Python dependencies. Users of Claude Code,
Codex, or an ordinary terminal run the same commands. A live Anthropic run still
requires `ANTHROPIC_API_KEY`; the launcher reads the configured provider key.
The generated HTML and `.xlsx` files can be opened independently of Python.

### Where files go

Local samples write **`reports/<sample>/report.html`**. Open that file to see
results; no separate render command is needed. For example:

```sh
uv run python -m agent_runtime --sample simple_chat
open reports/simple_chat/report.html
```

Rerunning a sample replaces its previous generated report bundle, including stale
Excel exports. There are no dates or run IDs in folder names. Other samples keep
their own results. Claims-context modes use `edit-with-patched-state` and
`edit-with-reloaded-state`. Browser turns replace the latest report for that sample.
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
| `samples/<sample>/sample.py` | Discovery metadata, authored prompts, and scripted model factories |
| `.cache/lg-report/prices/` and `.cache/lg-report/fx/` | Downloaded, reusable reference data; created dynamically, not execution reports |
| `data/rag/` and `.cache/lg-report/rag/` | Optional downloaded corpus/index archives and the restored retrieval index |

The samples keep their catalog, default caches, and `.env` relative to the
repository/sample even when launched elsewhere. Output files go to your working
directory. `prices.json` is a per-run snapshot of reference data, whereas
`models.json` and the caches are reusable inputs.

The samples capture message and tool content by default. `--metadata-only` omits those payloads. Exception messages are omitted; exception types identify failures. Credentials and ad hoc runs are ignored by Git; stable sample reports and evidence are included. Reporting uses a local exporter and disables inherited LangSmith tracing for the invocation; live prompts still go to the selected model provider.

## Report commands

The sample has already produced HTML. Use these commands only to rebuild saved
outputs; these commands do not run the agent:

```sh
uv run lg-report render reports/simple_chat/run.json
uv run lg-report normalize reports/simple_chat/spans.jsonl --static
uv run lg-report render reports/simple_chat/run.json
```

### Adaptive user for live tests

Live runs use **GPT-6 Luna at high effort** as the user agent by default,
through the existing Codex adapter:

```sh
uv run python -m agent_runtime --sample simple_chat --demo --live --user-turns 3 --env-file .env.local --prices models.json --out reports/adaptive-chat
```

Set `LG_USER_MODEL` in your environment or sample configuration, or use
`--user-model MODEL`, to choose a
different model; `--user-provider` / `LG_USER_PROVIDER` selects any registered
provider, including OpenAI, Anthropic, Copilot, or Codex. The user provider defaults
to Codex independently of the model being tested, so changing `LG_MODEL` or
`LG_PROVIDER` does not change the user persona's model. API user providers require
their normal credentials. `LG_USER_EFFORT` defaults to `high`;
`LG_USER_MAX_TOKENS` is an optional API-only output limit.

Choose the user agent's adapter with `--user-provider codex`,
`--user-provider copilot`, or `--user-provider openai` (also `anthropic`). Use
`--user-model MODEL_CODE` for a model available through that provider. Codex and
Copilot use their existing login; OpenAI and Anthropic require their normal API
key. The assistant's provider and model remain independently configurable.


The first authored request seeds the conversation verbatim, preserving structured
inputs. A generic user-role prompt loads the sample's canned user conversation
and clarification facts as guidance, without hardcoded sample facts or canned
assistant answers. Follow-ups and clarification replies adapt to actual model
output. `--user-turns` / `LG_USER_MAX_TURNS` bounds the conversation (default 3,
including the seed); this is a safety cap, not proof of success.

Each sample defines an optional nonempty `goal` at the start of its `SAMPLE`
configuration, before the conversation:

```python
SAMPLE = {
    "goal": "Explain the workflow steps and how tool observations inform an answer.",
    "id": "simple_chat",
    # Other discovery metadata follows.
}
```

The goal defines when the user model can stop: it must explain which conditions
were satisfied using the actual conversation. **With a goal, there is no minimum
turn count**; one complete answer can suffice. Without a goal, the sample's
number of authored user requests is the minimum and its conversation objectives
must be covered. The maximum always wins; an unmet goal at that limit is recorded
as `turn_limit`, not success. Increase `--user-turns` for longer scenarios.
The user model returns a JSON completion decision with a reason; invalid decisions
fail visibly. Full and comparison reports show the goal, stop status, and explanation
when content capture is enabled. These are user-model assessments, not independent
correctness verdicts.

Structured approval and clarification lessons use one initial request, with adaptive clarifications within
that turn. Clarifications are limited to ten answers per turn. Approval decisions retain the existing human or explicit
scripted test policy. Static runs stay deterministic. Select `--client static` for fixed user prompts
against a live assistant, or `--client console` / `--client angular` for human chat.
`--demo` requires an unattended client; it does not select model behavior.

For model comparisons, set the same `LG_USER_MODEL` and `LG_USER_PROVIDER` for all
runs. Adaptive prompts may diverge because assistant answers differ. User-model
calls are tagged in `run.json` and included in full-report totals. The comparison
shows their estimated cost separately, keeping tested-assistant costs, context,
and response columns separate from test-input generation. Failures remain visible;
there is no fallback to canned input or another model.

### Optional QA judge

Open the saved [QA example](reports/qa_evaluation/report.html) to inspect a live
GPT-6 Luna execution assessed by Sol 6. Optional QA evaluates the completed
execution with one independent judge:

```bash
uv run python -m agent_runtime --sample simple_chat --demo --live --qa --env-file .env.local
uv run python scripts/run_samples.py --qa
```

QA is off by default. `--qa` or `LG_QA=true` enables it; `--no-qa` disables
it. The default judge is Codex `gpt-6-sol` (Sol 6), high effort, using the
existing Codex login. `--qa-model` / `LG_QA_MODEL` and `--qa-provider` /
`LG_QA_PROVIDER` select one independent judge without changing the assistant
or user model. `LG_QA_EFFORT` and the API-only `LG_QA_MAX_TOKENS` control its
reasoning and output budget. Other providers need their usual credentials.

The judge first creates a task-specific rubric from the goal, before seeing
candidate results. Goal checks are binary; quality checks define full, half,
and zero credit. Each dimension has 100 available points. During evaluation,
the judge supplies criterion outcomes and evidence IDs; the application assigns
the points. Fulfilling every goal check always earns 100 for goal achievement.
The frozen rubric also defines numerical speed and cost anchors, so those
scores are calculated from measurements rather than individual opinions.
Weights remain goal 40%, quality 30%, speed 15%, and cost 15%.

Comparison runs generate one `qa-rubric.json` and pass it to every evaluation.
`LG_QA_RUBRIC` can supply the same saved criteria to separate launches; its goal
must match. Standalone judges create criteria on first use and reuse them for
subsequent turns with that goal. Expand **Scoring criteria** in the report to
inspect the common checks and formulas. Unknown dimensions remain unscored;
partial overall scores reweight available dimensions and display coverage.
Speed and cost include only Agent model calls. Static, simulated, and
metadata-only execution never constructs or calls the judge.

Saved `run.json`, HTML, comparison reports, and Excel retain the assessment.
Judge usage, elapsed time, and estimated cost are recorded separately from
execution totals. A judge error is visible without changing the sample status;
rendering saved reports never calls the judge again. Metadata-only runs skip QA
without sending content. The judge receives all captured events and their complete content. If the
serialized evidence exceeds 200,000 characters, QA skips the run without a model
call or score; it never clips histories or answers to fit. Browser QA
assesses each saved turn using its captured history and the selected sample goal.

### Compare models in one HTML report

Open the saved [live model comparison](reports/model_comparison/report.html) to
compare GPT-5.5 and GPT-5.6 Luna through both OpenAI and Codex, plus GPT-6 Sol
through Codex, on the same initial simple-chat prompt, with GPT-6 Luna generating adaptive user follow-ups. The report uses the single-model cost/context diagram with all models plotted together on
shared axes, then aligns responses side by side by recorded turn. Full-report
links retain the execution details and original pricing evidence. Without turn
metadata, alignment is explicitly by request order; unequal prompts remain visible.

Agent construction uses the standard LangGraph-backed `create_agent` with only
explicit tools and middleware. Simple chat has no tools on either API or CLI
routes. The shell and delegation lessons opt into the middleware they need.

The QA comparison table aligns overall and constituent scores, coverage, judge
identity, assessed goals, and reasons across the saved runs. Expand a score's
Reason to inspect its evidence. Missing, failed, or skipped QA remains explicit.
The execution measurements and cost chart cover Agent calls only. Judge cost
is omitted from the comparison. The shared criteria appear once above the table;
comparisons with differing saved criteria identify that mismatch explicitly.
Regenerating HTML only reads saved assessments and never calls a judge.

Supply the provider/model choices to one command. The runner applies medium
Agent effort without an explicit OpenAI output cap, a Codex GPT-6 Luna user at
high effort, and optional Sol 6 QA to every trial. It runs models sequentially
and builds the comparison automatically:

```sh
uv run python scripts/run_model_comparison.py --models openai:gpt-5.5 codex:gpt-5.5 openai:gpt-5.6-luna codex:gpt-5.6-luna codex:gpt-6-sol --qa --out outputs/five-model-comparison
```

`--effort`, `--sample`, `--user-model`, `--user-provider`, `--user-turns`,
`--qa-model`, and `--qa-provider` apply across the list. QA is optional and disabled
unless `--qa` is supplied. OpenAI uses its API credentials; Codex uses the installed
CLI and existing login. Set `LG_CODEX_MODEL_CATALOG=~/.codex/models_cache.json`
for the minimal Codex profile used by the refreshed reports; see
[Codex payload inspection](docs/codex-models.md#request-payload-inspection).
The runner requires unused model folders for fresh runs
and preserves previous measurements; choose a new `--out` for another trial.
To reassess existing executions, copy the selected bundles to a new directory
and use `--rescore --out DIRECTORY` with the same model list. This generates one
new shared rubric and calls only the judge, preserving the Agent traces and
measurements. `--saved-only` remains completely offline.

To rebuild the saved five-model example without making model calls:

```sh
uv run python scripts/run_model_comparison.py --models openai:gpt-5.5 codex:gpt-5.5 openai:gpt-5.6-luna codex:gpt-5.6-luna codex:gpt-6-sol --saved-only --out reports/model_comparison --title "Simple chat: API and Codex comparison"
```

The table includes Agent-only effective output rate and summed model-call
time per turn. These exclude user-agent and QA calls but include transport and
startup; they do not isolate pure decoding time. The recordings are single
observations with adaptive follow-ups and different answer lengths, not a
controlled provider-latency benchmark. Codex token usage includes harness context;
its estimated cost uses the corresponding OpenAI API rates, not subscription fees.

`lg-report compare RUN1 RUN2 ... --out comparison.html` remains available for
combining arbitrary saved runs without launching models.

The default output is `./comparison.html`; its parent directory must exist.
Each input requires an adjacent `prices.json`. Comparison preserves each run's
saved prices and FX; `--fx-file` is only supported for single-run rendering.
Missing usage or pricing stays explicit. Mixed-assistant runs list their tested
models and combined assistant totals; user-model input-generation costs are
reported separately. Repeated run or span IDs across recordings remain separate.
Elapsed time includes gaps between turns, so it is not a model-latency benchmark.
Use matching prompts, workflows, tools, and settings when comparing models;
the report does not automatically establish equivalent workloads or rank quality.

The architecture adds only a reporting projection and HTML template:
`reporting.compare` consumes existing `Run`/`Prices` records and calls the shared
`pricing.summarize` function for each run. Execution, capture, the `run.json`
schema, single-run HTML, and Excel accounting keep their existing contracts.

### Rebuild a retained run

For a retained run created with `--out reports/first-chat`:

```sh
uv run lg-report render reports/first-chat/run.json
uv run lg-report normalize reports/first-chat/spans.jsonl --static --title "Imported run"
```

Input filenames are optional; the defaults are `run.json` and `spans.jsonl` in
the working directory. Output defaults to `report.html` or `run.json` **beside
the input file**. `--out PATH` overrides the output filename. These commands
replace their output file.

`render` reads the adjacent `prices.json`; `--prices models.json` explicitly
selects another table. It reads the shared saved exchange rate without a network lookup. `normalize`
accepts this application's OTel SDK JSONL format, not arbitrary OTLP collector
JSON. Use `--static` only for simulated traces; omit it for live traces.
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

Run context calibration explicitly when you want to verify the configured models:

```bash
uv run python scripts/calibrate_models.py
# Or select a different catalog:
uv run python scripts/calibrate_models.py --config path/to/models.json
```

The command reads `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from the environment
or `.env`, checks every real entry in the catalog, and saves a `calibrations`
section in that same file. Demo entries use their declared model basis. It checks
model visibility with the key and obtains context capacity from OpenAI's official
model page or Anthropic's model metadata. It makes no generation requests and
does not empirically test the maximum prompt size or guarantee inference access.
Keys and raw error responses are never stored. Unsupported providers, missing
keys, and failed lookups are recorded as unavailable; the command exits nonzero
if any real model fails while still saving successful checks. Pricing fields and
aliases are preserved.

Calibration never runs at startup, during requests, or during report rendering.
New runs retain the saved calibration in their `prices.json` snapshot. An explicit
failed check makes capacity unavailable instead of reverting to the hardcoded
lookup; models with no calibration yet retain the legacy lookup. Existing saved
reports keep their original snapshot. To apply a calibrated catalog when
re-rendering an existing run, supply it explicitly with `--prices models.json`.
Capacity is the published total, not remaining room after input and output.

`models.json` contains exact `provider:model` keys and USD rates per million tokens. Costs use `Decimal`. Cache categories partition input; reasoning partitions output and uses the output rate. The app's cache TTL is five minutes. Unknown models or unpriced categories produce an incomplete known subtotal. At sample startup, catalog prices are refreshed from official provider pages. Before a new report is saved, any OpenAI, Anthropic, or Copilot model observed in the trace but missing from the snapshot is looked up automatically, including models used by subagents. Discovery validates the exact model identity and published pricing columns; dated IDs require matching provider identity evidence. Successful lookups and source text are saved under `.cache/lg-report/prices` by date and reused that day (`LG_PRICES_CACHE` overrides the directory). Copilot uses GitHub's own published token prices, including long-context tiers, to estimate usage before plan allowances. Codex uses the same model's OpenAI API rates as an estimated usage cost; the exact `codex:<model>` to `openai:<model>` alias is saved with the report. Explicit Codex tariffs or aliases take precedence. Discovered rates are included in the run's `prices.json`; the shared `models.json` seed is not rewritten. Demo rates explicitly follow GPT-5.6 Luna. `--prices` or `LG_PRICES` supplies an authoritative file and disables price fetching. Failed lookups retain the last verified prices and dates, with a warning in the report. A missing model whose lookup fails remains explicitly unpriced; failed discoveries retry on a fresh launch. Unsupported providers or page formats require a supplied file. Saved reports keep their exact `prices.json` snapshot, and `render` does not reprice historical runs automatically. Estimates exclude embedding calls, tool fees, infrastructure, taxes, and discounts.

The bundled catalog was verified on **2026-10-05** and includes 26 provider/model entries plus the existing demo tariff:

| Provider | Prepopulated selection |
| --- | --- |
| OpenAI (7) | GPT-6.1 Sol; GPT-6 Astra, Sol, and Luna; existing GPT-5.5 and GPT-5.6 Sol/Luna |
| Anthropic (8) | Claude Opus 5.5, Sonnet 5.5, Sonnet 4.6, Haiku 4.5 (canonical and dated IDs); existing Sonnet 5, Opus 4.8, and Fable 5.1 |
| Copilot (11) | GPT-6.1 Sol; GPT-6 Astra/Sol/Luna; GPT-5.4 and GPT-5.3-Codex; Claude Opus 5.5, Sonnet 5.5/4.6, and Haiku 4.5; Gemini 3.8 Flash |

These entries provide pricing references; they do not change the selected model or establish account access or context capacity.

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
uv run python -m reporting.export_excel .cache/excel/data.json outputs/lg-report-excel/report.xlsx
```

XlsxWriter writes formulas with cached results from the saved accounting data.
Excel recalculates them when you edit the execution count, tariffs, or exchange
rate. Unknown costs stay explicit. Export does not generate PNG workbook previews;
open the workbook to inspect its layout. Node.js is needed only to rebuild the Angular frontend.

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

New QA speed scores use `50*r/(r+R) + 50*T/(t+T)`, where `r` is reported
Agent output tokens per model-call second and `t` is model seconds per turn.
The judge selects the fixed half-credit anchors `R` and `T` from the task before
seeing results. Cost uses `100*C/(cost+C)`, with `C` the rubric's total Agent USD
cost at half credit. These task calibration choices are saved, shared and shown
in the report. They are not user budgets or provider service guarantees.
Missing measurements remain unscored. Legacy assessments retain their original
scoring basis until explicitly reassessed.

New OpenAI API runs stream responses to record client time to first output;
turn time uses the complete model call. Codex runs save internal OTLP timing
for time to first token only. Both use shared turn-time, first-token, and
output-rate rows, with coverage labels for partial capture. API first-token
timing is observed at the client; Codex supplies it internally and may include
hidden reasoning. These first-token boundaries differ. Extra backend diagnostic metrics are not retained.
The cost calculation recap after the graph reconciles token counts,
per-million rates, cache discounts, assistant totals, and saved FX conversion.
New QA resource assessments exclude simulated-user and judge calls. Shared FX
is displayed once; differing saved FX rates remain attributable to their runs.
