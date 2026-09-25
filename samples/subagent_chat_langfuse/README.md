<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Parent and subagent — Langfuse trace

## Purpose

Learn how a parent agent's `task` delegation becomes nested observations in
Langfuse. This is the same runnable workflow as the
[local-report sample](../subagent_chat/README.md): the parent delegates, the
specialist uses a local echo tool, and the parent answers from its summary.

The graph is shared intentionally. Comparing tracing systems should not introduce
a second version of the workflow or simulated token calculations. This directory
owns its catalog metadata; `agent_runtime.workflows.subagent_chat` composes the shared agents, and
`subagent_chat/sample.py` owns the conversation history from which the factory
extracts each caller's responses and scripted client prompts.

## Run

From the repository root:

```sh
uv sync --locked
uv run python -m agent_runtime --sample subagent_chat_langfuse --env-file .env.local --public-trace --demo
```

For the local Langfuse project `lg-report-dev`, set `LANGFUSE_BASE_URL` to
`http://localhost:3001` in the root `.env.local` and supply its project keys.
Docker/Langfuse must be running; operating instructions are in
`/Users/martinbechard/dev/langfuse-local/LOCAL-SETUP.md`.

For a fresh installation, copy the repository root `.env.example` to `.env.local`
if that file does not already exist, and set
`LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`. These must
belong to the chosen Langfuse project. Authentication is checked before invoking
any model. Shell environment variables override `.env.local`.

`--demo` uses simulated LLMs but sends real traces. To use real models, set
`LG_PROVIDER`, `LG_MODEL`, and the provider key in the same `.env.local`, then replace
`--demo` with `--live --client static` in the command. Both agents use that provider configuration; calls are
billable. Every run also produces the local `run.json`, `spans.jsonl`,
`prices.json`, and `report.html` bundle in `reports/subagent_chat_langfuse/`
(or `--out`). Excel remains a separate export. Both recorders observe one
execution; the workflow is not run twice.

## Tracing boundary

`sample.py` declares the shared delegation workflow and Langfuse tracing.
`execute_conversation` owns the same graph, execution, local-report, and cleanup sequence
for every console sample. `LangfuseCapture` handles configuration, project
authentication, the official callback, and shutdown; `conversation_trace` adds
trace visibility and turn spans. It also serves the
simple-chat Langfuse sample, so neither application duplicates this plumbing.

The handler is attached once at the parent invocation. DeepAgents propagates it
through `task` into the specialist. Attaching a second handler to the specialist
would risk duplicate observations and misleading totals.

## Expected trace

```text
subagent-chat-langfuse
  Turn 1
    delegating-parent
      model → parent generation (task request)
      tools → task
        isolated-subagent
          model → specialist generation (echo request)
          tools → echo_tool
          model → specialist generation (summary)
      model → parent generation (final answer)
```

There are four model generations and two tool executions. Inspect the specialist's
first request: it receives the delegated assignment, not the entire parent
history. Inspect the parent's final request: it receives the specialist summary,
not the specialist's internal tool exchange. Each agent has its own simulated
cache history, so the specialist starts with zero cache read.

With `--public-trace`, the returned URL opens without a browser login. Anyone
able to reach the configured server and holding the link can view captured
content. Without that flag, traces remain private. Our local server is bound to
localhost. API credentials and administrative login are still required for their
respective uses.

The fictitious `scripted-chat` model has no automatic price from our local
`models.json` in Langfuse. Token counts are recorded; configure a custom model
price in Langfuse if illustrative costs are needed. Missing cost is not zero.
Live usage comes from the provider. Our HTML/Excel import from Langfuse remains
a separate feature, not an automatic part of this sample.

## Verification

```sh
uv run pytest tests/test_subagent_sample.py tests/test_langfuse_sample.py -q
```

Tests cover nested task ancestry, isolated child context, summary-only return,
independent cache histories, and local model-cost totals. Langfuse tests use the
real SDK/callback with an in-memory exporter; the configured local server run
additionally checks persisted observations.

Reference: [DeepAgents subagents](https://docs.langchain.com/oss/python/deepagents/subagents).

The shared runtime uses a ConsoleClient with a ScriptPrompter through the common Conversation
loop. This sample uses the shared subagent workflow. Traces are private unless
`--public-trace` is supplied; public links expose captured content.

Use `--client console --live` for interactive prompts and text attachments.
