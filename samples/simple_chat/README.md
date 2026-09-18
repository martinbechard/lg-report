# Simple chat — conversation context

## Purpose

Learn how a second user turn differs from a fresh conversation. The second model
request retains the system instructions, the first user prompt, and the first
assistant response, then adds the new user prompt. This is the baseline for
understanding token and cost growth before introducing tools.

## Run

From the repository root:

```bash
uv sync
uv run python -m samples.simple_chat.app
```

The command prints its HTML report path and writes `spans.jsonl`, `run.json`,
`prices.json`, and `report.html` to a fresh directory under `reports/simple_chat/`.
Use `--out reports/my-chat-run` to name a new run directory.

To use a real model:

```bash
cp samples/simple_chat/.env.example samples/simple_chat/.env
# Edit that .env and supply the selected provider's key.
uv run python -m samples.simple_chat.app --live
```

The model adapter is the only part that changes between offline and live modes.
Live mode requires a valid key and incurs provider charges. Default mode makes
no provider request; the shared daily exchange-rate lookup may access the network.

## Code and execution flow

- `app.py` defines `SYSTEM_PROMPT`, `USER_PROMPTS`, `build_agent(model)`, and `main()`.
- `simulation.py` supplies two distinct assistant responses, not token totals.
- `build_agent()` uses `create_deep_agent()`, which returns a compiled LangGraph
  graph. Framework middleware remains visible in the execution tree.
- `execute()` attaches reporting around the graph invocation. The shared
  `ConversationAgent` passes the previous messages into the next user turn.

Expected offline sequence: **user → model response → user → model response**.
There are two LLM requests, labelled R1 and R2, and no tool execution.

DeepAgents still supplies built-in tool definitions; these consume input context
even when no tools are called. The system prompt tells the model to answer
directly. `StateBackend` keeps built-in file operations in graph state rather
than the developer's filesystem. This sample is not a bare tool-free model API.

## What to inspect

In R1, expand the fresh-input components and identify tool definitions, system
instructions, and the user prompt. In R2, compare cached history with the new
user prompt. Trace those counts through the cost chart and execution tree.
The run contains two turns; it must not treat the first answer as the final
answer for the entire conversation.

Try editing `USER_PROMPTS` and the matching scripted responses. Their message
lengths change the simulator's counts automatically. A real model may choose a
different answer length and cache behavior; those reported counts are retained.

See the [sample catalog](../README.md) for shared configuration, simulation
assumptions, and the boundary between model execution and report generation.
