<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Tool chat — request, observation, follow-up

## Purpose

Learn the difference between an LLM requesting a tool and the tool executing.
The LLM's tool-call JSON is output. The local tool function returns an
observation, which becomes input to the next LLM request. The tool does not
make its own LLM call.

## Run

From the repository root:

```bash
uv sync
uv run python -m agent_runtime --sample tool_chat --demo
```

Each run writes raw spans, normalized data, the pricing snapshot, and HTML under
`reports/tool_chat/`, replacing the previous run. Use `--out reports/my-tool-run` to select
another reusable directory.

For a real model, copy `samples/tool_chat/.env.example` to
`samples/tool_chat/.env`, provide the selected key, and run:

```bash
uv run python -m agent_runtime --sample tool_chat --live
```

`--demo` runs use the simulator, so they do not call a model provider. FX reads the shared
`exchange-rate.json` without a network lookup. Live mode incurs model
charges and is not required to follow the scripted call sequence.

## Code and execution flow

- `sample.json` declares the workflow and script; the shared launcher selects client and recorder.
- `src/agent_runtime/workflows/tool_chat.py` composes the participating agents.
- `src/agent_runtime/agents/reference_chat_agent.py` owns the agent instructions and registration.
- `src/agent_runtime/tools/echo_tool.py` owns the local echo tool.
- `scripted_run.py` scripts four model responses. It does **not** fake tool
  execution: LangGraph dispatches the registered Python function.
- Shared reporting wraps the application; no report-generation code is inside
  the tool or graph definition.

Each of the two user turns follows:

1. The model returns a `echo_tool` tool request.
2. LangGraph invokes the function and appends a tool-result message.
3. The model consumes the expanded history and returns an answer.

The offline report therefore has four model calls and two tool calls. Tool-call
IDs match each request with its observation; they are not separate LLM requests.
Assistant text and tool-call JSON can coexist in the same model response.

## What to inspect

R1's output includes tool-call JSON. The following tool span contains its
arguments and result. R2's fresh input contains the observation, while the
simulator reuses prior conversation as cached history. Compare that sequence
with the second turn, where R3/R4 continue the request numbering.

`echo_tool(text: str)` returns `Your input was: ` followed by the supplied
string unchanged. For example, `echo_tool` with `text="ReAct"` returns
`Your input was: ReAct`. It repeats input and supplies no independent evidence.
It performs no search, network access, or model call. Tool execution has no external service fee in this sample; real
tools may have costs outside model-token accounting.

To experiment, change the echo implementation or add a tool. Register it
in `build_agent()` and update the scripted response's tool name and arguments.
The request/result linkage and captured graph should still reconcile.

See the [sample catalog](../README.md) for common configuration and simulation
assumptions. The reporting library never infers real cache usage from history.

## Interactive client

Replace `--demo` with `--client console --live` in the launch command after configuring this sample's
`.env`. The shared console accepts prompts and `/attach PATH` text files, `/send`,
and `/quit`. Both clients use the same workflow and retained conversation history.
See [component and sequence diagrams](../../docs/chat-composition.md).
