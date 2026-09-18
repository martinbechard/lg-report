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
uv run python -m samples.subagent_chat.app
```

The command prints the HTML path and creates `spans.jsonl`, `run.json`,
`prices.json`, and `report.html` under `reports/subagent_chat/<run>/`.
`--out` selects a new directory; existing directories are rejected.
`--fx-file` supplies a local USD/EUR rate; otherwise the shared daily lookup/cache
is used. No Langfuse service or credentials are involved.

For a real model:

```sh
cp samples/subagent_chat/.env.example samples/subagent_chat/.env
# Configure LG_PROVIDER, LG_MODEL, and the selected provider's API key.
uv run python -m samples.subagent_chat.app --live
```

The same provider/model configuration is used for both agents, with separate
adapter instances. Live mode incurs provider charges and can produce a different
number of calls. Environment variables override the sample's `.env`.

## Read the code

- `app.py` selects models, client, and recorder.
- `src/lg_report/workflows/subagent_chat.py` composes the participating agents.
- `src/lg_report/agents/delegating_parent.py` owns the agent instructions and registration.
- `src/lg_report/tools/workflow_reference.py` owns the callable evidence tools.
- `src/lg_report/agents/workflow_specialist.py` owns the child role and lookup tool.
- `test_case.py` owns two independent scripted models. Scripts specify model
  responses and tool requests; they do not execute tools or calculate totals.
- `workflow_reference` is reused from the tool-chat lesson. It is a local fixture
  lookup, not a network search or a vector RAG pipeline.
- `lg_report.platform.sample_runtime` handles configuration and attaches the local recorder.

The specialist dictionary explicitly supplies its own system prompt, model, and
lookup tool. The parent does not have that lookup tool. Its generated `task`
request selects `workflow-specialist` and supplies a self-contained assignment.
DeepAgents performs the invocation and returns the specialist's final message as
the parent's tool result. The application never calls the specialist manually.

`StateBackend` keeps framework file operations in memory. DeepAgents also exposes
its standard built-in tools/default delegation capabilities; the scripted run
uses only the named specialist and the reference lookup.

## Expected sequence

One user turn contains **four model requests and two tool executions**:

1. Parent R1 emits a `task` request for `workflow-specialist`.
2. Inside that task, specialist R2 requests `workflow_reference`.
3. The lookup executes; specialist R3 consumes the result and returns a summary.
4. Parent R4 consumes that summary and produces the final answer.

In the execution tree, expand `task`: the specialist's graph, models, and lookup
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
are illustrative JSON words/punctuation, not provider tokenizer counts. Actual
provider usage is retained in live mode. All four model calls contribute to the
turn cost; `task` and the local lookup have no additional model-token charge.

Try changing `DELEGATED_TASK` and `SPECIALIST_SUMMARY` in `test_case.py`. Inspect
which agent's input grows and where the summary enters the parent's context.
Keep the tool request's `subagent_type` aligned with the registered specialist.

The [Langfuse variant](../subagent_chat_langfuse/README.md) reuses this exact graph
and simulation so you can compare the same workflow through two tracing systems.

## Interactive client

Add `--client console --live` to the launch command after configuring this sample's
`.env`. The shared console accepts prompts and `/attach PATH` text files, `/send`,
and `/quit`. Both clients use the same workflow and retained conversation history.
See [component and sequence diagrams](../../docs/chat-composition.md).
