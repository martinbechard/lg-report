<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Parent and subagent — local report

## Purpose

Learn what delegation actually does to execution, context, and cost. The parent
asks a specialist for a bounded piece of work; the specialist calls a real local
tool and returns a summary. The parent then produces the user-facing answer.
Both are real DeepAgents/LangGraph agents, even when the models are simulated.

## Run

From the repository root:

```sh
uv sync --locked
uv run python -m agent_runtime --sample subagent_chat --demo
```

The command prints the HTML path and creates `spans.jsonl`, `run.json`,
`prices.json`, and `report.html` in `reports/subagent_chat/`, replacing the previous default run.
`--out` selects another directory; reruns replace its generated reports.
`--fx-file` supplies a local USD/EUR rate; otherwise the saved shared
`exchange-rate.json` is used without network access. No Langfuse service or credentials are involved.

For a real model:

```sh
test -f .env.local || cp .env.example .env.local
# Configure LG_PROVIDER, LG_MODEL, and the selected provider's API key.
uv run python -m agent_runtime --sample subagent_chat --live --env-file .env.local
```

The same provider/model configuration is used for both agents, with separate
adapter instances. Live mode incurs provider charges and can produce a different
number of calls. Shell environment variables override `.env.local`.

## Read the code

- `sample.py` declares the sample ID, display name, and description. The ID
  determines the workflow and script module paths. The shared launcher selects clients and recording.
- `src/agent_runtime/workflows/subagent_chat.py` composes the participating agents.
- `src/agent_runtime/agents/delegating_parent.py` owns the agent instructions and registration.
- `src/agent_runtime/tools/echo_tool.py` owns the local echo tool.
- `src/agent_runtime/agents/isolated_subagent.py` owns the child role and echo tool.
- `sample.py` describes one conversation in execution order. `client` entries
  supply scripted prompts; `ai` entries supply the workflow-created model;
  `isolated-subagent` entries supply the model created by that agent.
  Tool entries document expected observations; actual tools still run.
- `src/agent_runtime/harness/model_factory.py` provides
  `build_model(model_name=None, *, caller)`. Omit the name to use `LG_MODEL`.
  The factory creates a fresh provider or simulated model on each call. Only
  the harness chooses the mode; workflows and agents do not import fixtures.
- `echo_tool` is reused from the tool-chat lesson. It returns
  `Your input was: ` followed by the supplied `text` unchanged, with no network
  or model call. The echo adds no independent factual evidence.
- `agent_runtime.harness.sample_catalog.SampleCatalog` discovers metadata and
  establishes the model factory scope. `settings.prepare_sample` resolves
  shared launch inputs; `main` selects the client or listener. `execute_conversation`
  executes a prepared console/scripted conversation with its recorder.
- The launcher creates `ConsoleClient()` directly and calls
  `configure_sample_script` from `harness/configure_sample_script.py` when authored
  requests are needed, attaching a `ScriptPrompter` to that client.

The workflow calls `build_model(caller="workflow")` and passes that model to the
parent. The specialist calls `build_model(caller="isolated-subagent")` itself.
Both calls use the same factory in live and simulated runs. Outside the runtime's
construction scope the factory defaults to real models; live errors propagate.
A model retains its mode and independent response cursor after the scope ends.
This example uses one workflow-created model and one agent-created model; it
does not infer separate response streams for multiple agents sharing one object.

The specialist dictionary explicitly supplies its own system prompt, model, and
echo tool. The parent does not have that echo tool. Its generated `task`
request selects `isolated-subagent` and supplies a self-contained assignment.
DeepAgents performs the invocation and returns the specialist's final message as
the parent's tool result. The application never calls the specialist manually.

`StateBackend` keeps framework file operations in memory. DeepAgents also exposes
its standard built-in tools/default delegation capabilities; the scripted run
uses only the named specialist and the echo tool.

## Expected sequence

One user turn contains **four model requests and two tool executions**:

1. Parent R1 emits a `task` request for `isolated-subagent`.
2. Inside that task, specialist R2 requests `echo_tool`.
3. The echo executes; specialist R3 consumes the result and returns a summary.
4. Parent R4 consumes that summary and produces the final answer.

In the execution tree, expand `task`: the specialist's graph, models, and echo tool
must be descendants of it. The conversation and chart include all four model
calls. Parent/subtree rows already contain descendant costs; do not add them
again to model rows.

## What this teaches

The specialist's first request contains its own system instructions and the
**delegated task**, not the original user conversation. Its intermediate tool
exchange remains in its context. The parent receives only the final summary.
Therefore there are two separate context histories, not one continuously growing
history shared by every model call in the report.

Each simulated agent has its own context meter and response cursor. Its first
request has no cached history; its second reuses its own previous context. Counts
use the shared tiktoken `o200k_base` estimator over canonical JSON, not a provider's
exact request envelope. Actual provider usage is retained in live mode. All four model calls contribute to the
turn cost; `task` and the local echo have no additional model-token charge.

Try changing the literal task description and specialist reply in `CONVERSATION`
in `sample.py` (and its matching task-result entry). Inspect
which agent's input grows and where the summary enters the parent's context.
Keep the tool request's `subagent_type` aligned with the registered specialist.

The [Langfuse variant](../subagent_chat_langfuse/README.md) reuses this exact graph
and simulation so you can compare the same workflow through two tracing systems.

## Interactive client

Replace `--demo` with `--client console --live --env-file .env.local` in the launch command after configuring the repository
`.env.local`. The shared console accepts prompts and `/attach PATH` text files, `/send`,
and `/quit`. Both clients use the same workflow and retained conversation history.
See [component and sequence diagrams](../../docs/chat-composition.md).
