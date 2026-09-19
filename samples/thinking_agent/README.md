<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Thinking agent — evidence, reasoning, verification

## Purpose

Learn how reasoning tokens can dominate cost even when the visible answer is
short. The task is to propose a latency improvement for a fictional service
under explicit constraints, then verify the plan before recommending it.
The report distinguishes reasoning from ordinary output and tool execution.

## Run

From the repository root:

```bash
uv sync
uv run python -m samples.thinking_agent.app
```

The report is written as `./report.html` in the current working directory together
with the raw trace, normalized run, and prices. `--out reports/my-investigation`
selects a new directory; existing runs are not overwritten.

For a live model, copy `samples/thinking_agent/.env.example` to
`samples/thinking_agent/.env`, supply a provider key, choose a model supporting
the desired reasoning settings, and run:

```bash
uv run python -m samples.thinking_agent.app --live
```

Do not assume the model accepts every effort setting. The default offline
`high` label is an annotation; live effort comes from the provider configuration.
The live model chooses its own calls and reasoning usage. These paths are not
tested against a paid provider merely by running the offline fixture tests.

## Code and execution flow

- `app.py` selects models, client, and recorder.
- `src/lg_report/workflows/thinking_agent.py` composes the participating agents.
- `src/lg_report/agents/investigation_agent.py` owns the agent instructions and registration.
- `src/lg_report/tools/service_evidence.py` owns the callable evidence tools.
- `test_case.py` owns the seven scripted LLM responses. It specifies the
  teaching reasoning counts and illustrative thinking text.

The deterministic sequence is:

| Request | Purpose | Next tool |
| --- | --- | --- |
| R1 | Inspect traffic | `inspect_service(traffic)` |
| R2 | Inspect database behavior | `inspect_service(database)` |
| R3 | Inspect constraints | `inspect_service(constraints)` |
| R4 | Evaluate the candidate; 12,000 reasoning tokens | `test_plan(load)` |
| R5 | Verify freshness; 600 reasoning tokens | `test_plan(freshness)` |
| R6 | Verify rollback; 600 reasoning tokens | `test_plan(rollback)` |
| R7 | Recommend the verified plan; 1,800 reasoning tokens | None |

There is one user turn, seven model calls, and six tool calls. The heavy
reasoning response is preceded by three tools and followed by three tools.

## What to inspect

Find R4 in both the conversation and stacked-cost chart. Its reasoning segment
uses the output-token rate. The visible output and tool-call JSON remain a
separate category. The report displays at most three italic lines of available
thinking text without generating a summary.

The fixture's thinking text is illustrative and is not a literal transcript of
12,000 tokens. Reasoning counts are deliberately specified for the lesson;
visible message counts come from the simulator's message meter. Hidden reasoning
is not appended to visible history or marked as cached response text. Real
providers may return reasoning text, summaries, redactions, or only token usage;
the sample must not invent text for a live response.

Try changing the reasoning count in `test_case.py` while leaving the visible
response unchanged. Its output cost should change, but the subsequent visible
context should not grow by that hidden reasoning count. Then change a tool
result and observe fresh-input/context growth instead.

See the [sample catalog](../README.md) for common configuration and the limits
of the cache simulation. The tools are demonstrations, not a production agent.

## Interactive client

Add `--client console --live` to the launch command after configuring this sample's
`.env`. The shared console accepts prompts and `/attach PATH` text files, `/send`,
and `/quit`. Both clients use the same workflow and retained conversation history.
See [component and sequence diagrams](../../docs/chat-composition.md).
