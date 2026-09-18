<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Simple chat with Langfuse

This is the same `agents/chat_agent.py` used by the local-report sample. It uses
the same clients and `platform.Conversation`; only the recording wrapper changes.
There is no second graph definition and no Langfuse-specific conversation loop.

Start with [How the parts fit together](../../docs/chat-composition.md) for the
Mermaid component and sequence diagrams, ownership table, and failure behavior.

## Configuration

```bash
cp samples/simple_chat_langfuse/.env.example samples/simple_chat_langfuse/.env
```

Set `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` for your
project. For the existing local installation, the endpoint is
`http://localhost:3001`; keys belong in the private `.env`, not source control.
Live model mode additionally needs the selected provider key and model settings.

## Static test case

```bash
uv run python -m samples.simple_chat_langfuse.app
```

This reuses `samples/simple_chat/test_case.py`: two user prompts and scripted
assistant responses. It makes no LLM provider call, but sends real traces to the
configured Langfuse endpoint. `--live` uses the same requests with a real model.

## Console chat

```bash
uv run python -m samples.simple_chat_langfuse.app --client console --live
```

Enter prompts at `User:`. `/attach PATH` queues a UTF-8 text file, `/send` sends
queued files without additional prompt text, and `/quit` ends the session. EOF
also ends normally. History and attachments persist across requests. The console
requires a real model because the static answers cannot answer arbitrary prompts.

Traces are **private by default**. Add `--public-trace` only when you want the
captured content visible through a public trace link. This includes attached file
text and applies to hosted as well as local Langfuse. API keys are still required.
No local HTML, Excel, pricing refresh, or trace JSON files are produced here.

## Execution and trace structure

1. `app.py` chooses models, the shared `workflows/simple_chat.py` workflow, and a client.
2. `platform.langfuse_runtime.launch` loads configuration and validates access
   before constructing the graph. It selects the static or console client.
3. `run_conversation` opens one root and attaches the official callback once.
4. `Conversation` requests each user turn, retains history, and invokes the graph.
   Its optional scope hook opens `Turn N`, records the result, and closes the span.
5. The callback records nested graph/model/tool observations. The wrapper flushes
   and shuts down the SDK before exit and prints the trace URL on success.

Inspect one root, two turn spans, and two model observations in the default test.
The second model request should include the first prompt and answer. Static and
console clients use identical history rules. Client waiting time belongs to the
root duration, not the turn duration.

Failures close turn/root spans as errors and still shut down the SDK. Empty sessions
are incomplete; approval interrupts stop the conversation and are marked interrupted.
This sample does not implement approval/resume. A printed URL is an identifier;
server ingestion may take a moment after SDK export finishes.

`tests/test_langfuse_sample.py` uses the actual SDK/callback with an in-memory
exporter to verify trace ancestry, usage, content, clients, and failure handling.
