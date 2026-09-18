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
uv run python -m samples.tool_chat.app
```

Each run writes raw spans, normalized data, the pricing snapshot, and HTML under
`reports/tool_chat/<timestamp>-<id>/`. Use `--out reports/my-tool-run` to select
a fresh directory.

For a real model, copy `samples/tool_chat/.env.example` to
`samples/tool_chat/.env`, provide the selected key, and run:

```bash
uv run python -m samples.tool_chat.app --live
```

Default runs use the simulator, so they do not call a model provider. The daily
FX lookup is separate and can be supplied from a file. Live mode incurs model
charges and is not required to follow the scripted call sequence.

## Code and execution flow

- `app.py` owns the graph and prompts. The system instruction asks for a reference
  lookup before answering, explaining why a tool is relevant to this task.
- `tools.py` owns `workflow_reference(topic)`. Its docstring is exposed to the
  model as the tool description; the implementation returns a local reference.
- `simulation.py` scripts four model responses. It does **not** fake tool
  execution: LangGraph dispatches the registered Python function.
- Shared reporting wraps the application; no report-generation code is inside
  the tool or graph definition.

Each of the two user turns follows:

1. The model returns a `workflow_reference` tool request.
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

The reference is intentionally small and local. This demonstrates a lookup
tool, **not** embeddings, vector retrieval, document permissions, or a complete
RAG system. Tool execution has no external service fee in this sample; real
tools may have costs outside model-token accounting.

To experiment, change the reference implementation or add a tool. Register it
in `build_agent()` and update the scripted response's tool name and arguments.
The request/result linkage and captured graph should still reconcile.

See the [sample catalog](../README.md) for common configuration and simulation
assumptions. The reporting library never infers real cache usage from history.
