# Simple chat with Langfuse

## Purpose

Learn where tracing attaches to a DeepAgents/LangGraph application. This variant
runs the same two-turn conversation as [simple chat](../simple_chat/README.md),
but sends execution traces through the **official Langfuse callback**. It does
not attach `TraceCapture`, call `record_run`, or generate local HTML/Excel files.
The result is a public trace you inspect in Langfuse without logging in.
The sample publishes every run, including live runs. Anyone who can reach the
configured server and has the link can view its contents. Our local instance
is bound to localhost; project administration and API ingestion still require
authentication.

The graph remains application code in `app.py`. Prompts and scripted responses
are reused from the baseline sample so the instrumentation is the meaningful
difference. The simulator supplies usage through the normal model message API;
it does not create Langfuse spans itself.

## This workstation

The local development instance is at http://localhost:3001, with project
`lg-report-dev`. Its Compose checkout and operational notes are in
`/Users/martinbechard/dev/langfuse-local/LOCAL-SETUP.md`. The sample
`.env` is configured for that instance; run the command below directly.
Local login credentials are kept in the Compose checkout's private `.env`,
not in this README.

## Configure and run

From the repository root:

```sh
uv sync --locked
cp samples/simple_chat_langfuse/.env.example samples/simple_chat_langfuse/.env
```

Create a project in your Langfuse instance and enter its public key, secret key,
and base URL in that `.env`. The example URL is Langfuse Cloud's EU endpoint.
For a local instance, set `LANGFUSE_BASE_URL=http://localhost:3000` and use keys
created in that local project. This sample does not provision a Langfuse server.

Run the application:

```sh
uv run python -m samples.simple_chat_langfuse.app
```

It checks project authentication, prints both answers, flushes pending spans,
and prints the trace URL. Langfuse ingestion is asynchronous: the trace may take
a moment to appear. A printed URL is not a server-side delivery confirmation;
check the UI and any exporter errors. Missing configuration or authentication
failure stops execution before model invocation. There is no local-capture fallback.

By default, the LLM is simulated, so no OpenAI/Anthropic key or paid model call is
needed. **Tracing is real:** messages, responses, and metadata are sent to your
configured Langfuse server. This is not a fully offline command. The application
uses no exchange-rate service, pricing file, or local report directory.

To call a real model, configure `LG_PROVIDER`, `LG_MODEL`, and its API key in the
same `.env`, then add `--live` to the command above. Only the selected provider's
key is required. Live calls incur provider charges. Shell variables take
precedence over `.env`; `--env-file` selects a different configuration file.

## Read the code

1. `build_agent()` constructs the graph. `StateBackend` keeps built-in file tools
   in memory. The prompt requests direct answers, although DeepAgents still
   supplies its built-in tool definitions to the model.
2. `main()` calls the shared `lg_report.langfuse_runtime.launch`, which loads
   configuration and verifies Langfuse access before selecting
   the model. `CallbackHandler` is selected explicitly for that project key.
3. The shared runtime's `run_conversation()` opens one parent trace and a span for each turn. Passing
   the handler in `graph.invoke(config=...)` captures nested graph/model calls.
4. Each new request includes the complete preceding message history. A shared
   trace groups execution; it does not store conversation state for the graph.
5. The shared runtime flushes on success and shuts down in `finally`, including failures,
   because this short-lived process must allow queued telemetry to finish.

Ambient LangSmith tracing is disabled for these invocations so this lesson has
one tracing destination. Errors from the graph propagate through Langfuse's
span contexts; the CLI prints an exception type without copying provider error
payloads to the terminal.

## What to inspect

Expect this hierarchy in simulated mode, with framework nodes between turns
and model generations:

```text
simple-chat-langfuse
  Turn 1
    chat-agent → model → generation
  Turn 2
    chat-agent → model → generation
```

There should be two model generations, distinct answers, and no tool execution.
Turn 2 includes the earlier user prompt and assistant response. Inspect its
input/cache usage and compare it with Turn 1. The callback preserves the model's
`report_effort`, purpose annotation, and the invocation's `report_turn` metadata.

Simulated counts measure message/tool-definition JSON, not provider tokenizer
output. They assume reuse of the completed conversation as cached context. In
live mode, provider-reported usage determines the counts. Langfuse normalizes
input into exclusive fresh/cache categories, so do not subtract cached tokens
from its fresh-input value again.

`scripted-chat` is a fictitious model. Langfuse does not receive our `models.json`
price table, so this sample does not promise a cost for it. Configure an explicit
custom model price in Langfuse if you want illustrative costs. Missing pricing
is not evidence of zero cost. Real model pricing depends on Langfuse's model
configuration. EUR conversion, our presentation, and workbook projections are
outside this tracing sample; a Langfuse-to-`Run` importer is not implemented here.

## Verification

```sh
uv run pytest tests/test_langfuse_sample.py -q
```

The tests run the real graph, Langfuse SDK, and official callback with an
in-memory span exporter. They verify parentage, two-turn history, token/cache
accounting, annotations, error spans, and configuration failures without a
server or model credentials. Hosted authentication, ingestion, and UI display
require running against your configured project.

References: [Langfuse DeepAgents integration](https://langfuse.com/integrations/frameworks/langchain-deepagents),
[LangChain callback](https://langfuse.com/integrations/frameworks/langchain),
[token accounting](https://langfuse.com/docs/observability/features/token-and-cost-tracking).
