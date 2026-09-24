<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Angular sample chat

A single chat window demonstrates the same sample factories used by the console.
The small menu selects a sample; the composer supports text attachments, Send,
and Stop. Tool calls appear inline with expandable arguments and results; a
received result is distinguished from a call still waiting for one. The response-mode dropdown beside the sample selects real LLM or scripted responses.
Changing mode starts a fresh chat. Model configuration remains on the server.

## Run

The compiled UI is checked into `frontend/dist/chat/`; running it needs no Node.js.
From the repository root:

```sh
uv sync --extra chat

uv run --extra chat python -m agent_runtime --sample simple_chat --live --client console
uv run --extra chat python -m agent_runtime --sample simple_chat --live --client angular
```

Open [the chat](http://127.0.0.1:8000). The sample's `--env-file`, `--prices`, and
`--out` options still apply; `--port` changes the server port. Use `--demo` for
scripted responses as the initial selection. The dropdown can change this mode
without restarting the server; real LLM mode needs configured provider credentials.

Every web launch exposes the whole discovered catalog. For a scripted initial session:

```sh
uv run --extra chat python -m agent_runtime --client angular
# Add --live --env-file samples/simple_chat/.env for free-form model conversations.
# Optional: --prices models.json --port 8000
```

The catalog includes all local samples, file approval, quote clarification, both
claims-context strategies, and the two Langfuse recording variants. Langfuse
selections require configured tracing credentials; they fail visibly rather than
silently disabling remote recording. Retrieval uses the existing index setup.

## Shared samples, standard streaming

`SampleCatalog.create_run(sample_id, live)` constructs the selected workflow
inside its model factory scope. Neither agents nor workflows import the frontend.
Both client paths use
`Request.content()` and `Attachment` to format user context.

```text
Console → ConsoleClient → AG-UI client event pump → LangGraphAgent → sample graph
Angular → AG-UI HttpAgent → HTTP/SSE → LangGraphAgent → sample graph
```

Both clients use LangGraphAgent and LangGraph's in-memory checkpointer for history,
including tool messages. The server accepts only the newest user request;
browser-supplied state and tool definitions cannot change the sample's policy.
MCP uses the existing agent factory inside its required async transport context.

[Deep Agents supports AG-UI through LangGraph](https://docs.langchain.com/oss/python/deepagents/ag-ui).
The official Python `LangGraphAgent` produces events, `EventEncoder` formats SSE,
and the official JavaScript `HttpAgent` parses events, accumulates messages and
aborts requests. There is no application-specific socket protocol or token
reducer. Model text streams as available. Scripted models emit their authored
text and tool argument chunks through the same native model streaming API. Subagent invocations have expandable conversations
attached to their spawning tool calls, including the delegated task, messages,
tools, results and lifecycle status. Nested delegations expand in the same way.
Earlier child conversations remain available after later turns. Review workflow
output is streamed as the standard adapter emits it, including intermediate text.

The parity test runs every catalog sample through the real console, including
`/attach`, and the SSE endpoint. It compares final user, assistant and tool
message semantics using fresh deterministic models. This verifies shared sample
behavior; independent live model runs can still produce different wording.

To add a compatible sample, declare its workflow and script in the folder's
`sample.py`. The catalog discovers it on startup without a central registration
edit. The parity test covers all discovered samples.

## Attachments and local state

Attach or drop UTF-8 files, as with console `/attach`. Text, Markdown, JSON, CSV,
and source code work; PDF/image extraction is not implemented. Files queue until
Send and can be removed. Live mode also accepts attachments without a prompt,
like console `/send`. The complete outgoing JSON request has an 8 MiB limit.
Attachment contents enter model history and grant no file-write permission.

New chat starts a fresh model and checkpoint. Stop aborts the HTTP stream and
cancels native async execution; an in-flight synchronous tool or provider request may still finish. A stopped or failed turn requires a new chat.
Already completed tool effects and provider billing cannot be undone. There is
no automatic retry. Inactive sessions expire after an hour when a new session is
created; browser cleanup requests release them sooner when possible.

The existing capture, normalizer, pricing and HTML renderer save one report
bundle at `reports/<sample>/report.html`, replacing the previous turn's output.
No dated or UUID report folders accumulate. Reports remain
available on disk without adding controls to the chat. The shared launcher
captures content by default; `--metadata-only` omits it from saved reports.
`run.json` remains independent of HTML and unknown usage/prices remain unknown.
Ambient hosted LangSmith tracing is disabled for browser runs. The catalog
launcher uses the shared price-loading policy; pass a snapshot with `--prices`
to avoid refreshing prices during startup.

The server binds to loopback and checks browser origins. This is a single-user
local demonstration application.

## Development and verification

Install a supported Node.js version (22.12+ or 24), then run
`npm --prefix frontend ci`. After frontend source changes, run
`npm --prefix frontend run build` and commit the complete `frontend/dist/chat/`
output with the source, including removed hashed assets and third-party licenses.


Run the Python server on port 8000, then `npm --prefix frontend start` for Angular
live reload at [localhost:4200](http://localhost:4200). The development proxy
forwards HTTP requests and SSE responses.

```sh
uv run --extra chat pytest tests/test_web_chat.py tests/test_simple_chat_clients.py
npm --prefix frontend run build
cd frontend
npx playwright install chromium
npm test
```

Browser tests cover real SSE conversations, attachment framing, sample selection,
reset, UTF-8 rejection and mobile layout. A gated HTTP fixture checks incremental
rendering and Stop before the stream finishes; a backend gated model separately
checks native graph cancellation and partial report capture. Screenshots and
traces remain ignored. These checks make no paid provider calls.

## Approval, clarification, and context

An interrupted run leaves its session usable. The UI presents the workflow's
approval proposal or clarification question and sends a native AG-UI resume
entry for each pending ID. New user messages are rejected while an interaction
is pending. Approve/reject/cancel do not repeat the preceding model call.

The file sample's catalog session has its own temporary output; a download link
appears after an approved write. New chat or session expiry removes that workspace.
Claims modes expose a context audit independently of their display transcript.
Workflow cancellation follows the graph's policy. Stop aborts the HTTP run and
requires a fresh chat; it neither approves a tool nor rolls back earlier effects.

The context meter previews retained context for the next request, excluding the
composer draft and newly queued attachments. It reads the server checkpoint after
the turn, so the final answer is included and managed claims purges are reflected.
Expand **View retained context** to inspect messages, instructions, and tool schemas.
Token counts and percentages are local estimates, not provider usage receipts;
bytes count the UTF-8 JSON representation including tool definitions. For normal
agent graphs, instructions and schemas come from the entry agent's first call in
the last turn, so the initial preview is unavailable and next-turn middleware can
still change it. Author/judge and quote workflows expose their own read-only input
projection to avoid mistaking display history for the next model's context.
Preview content stays in session memory and the local browser; report files still
follow the launcher's content-capture setting.
