<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Simple chat — an agent with interchangeable clients

## Purpose

Learn to separate an agent's behavior from its user and its test cases. The agent
receives messages at runtime; it does not know the test prompts, read terminal
input, or generate reports. The session retains history so later requests include
earlier user messages, attachments, assistant answers, and any tool exchanges.

## Files and responsibilities

The reusable code lives under `src/lg_report/`:

- **`workflows/simple_chat.py`** composes the single-agent workflow.
- **`agents/chat_agent.py`** defines the named agent, its system instructions,
  and its DeepAgents graph. It imports no client or test case.
- **`platform/conversation.py`** defines requests, text attachments, the client
  interface, and the session loop that retains history and turn metadata.
- **`platform/console_client.py`** accepts human prompts and text files.
- **`platform/static_client.py`** consumes requests supplied by any test case;
  it contains no predefined prompts or expected answers.
- **`platform/simulated_model.py`** and **`platform/demo_meter.py`** provide the
  generic offline model and context/token simulation.
- **`report/`** captures traces, prices usage, and exports HTML/Excel.
- **`tools/`** is the home for application tools. Simple chat defines none.

This sample directory contains **`app.py`** for component wiring and
**`test_case.py`** for the scenario's user prompts and prerecorded model answers,
plus this README and configuration example. A client simulator and a model
simulator serve different roles even when one test case configures both.

A future user-avatar agent can implement `receive()` and `respond(result)` from
`ChatClient`: after receiving an answer, it chooses the next request, or returns
`None` to end. Such an avatar would live in its own named file under `agents/`.
No avatar agent is implemented yet.

## Run the static test case

```bash
uv sync
uv run python -m samples.simple_chat.app
```

Default client: `static`. Default model: simulated. This executes a real graph
with two predefined user turns and two prerecorded responses, without provider
charges. Daily pricing/FX lookups may still access the network. Use `--prices`
and `--fx-file` to supply those references without lookups.

The graph still includes DeepAgents' built-in tool definitions, which consume
context even though its instructions ask it to answer without tools. StateBackend
keeps built-in file operations in graph state rather than the local filesystem.

## Console chat

```bash
cp samples/simple_chat/.env.example samples/simple_chat/.env
# Configure an OpenAI or Anthropic API key in that file.
uv run python -m samples.simple_chat.app --client console --live
```

Enter a prompt and read the response. Commands:

- `/attach /path/to/notes.txt` queues a UTF-8 text file for the next prompt.
  Paths may contain spaces; do not surround them with quotes.
- `/send` submits queued files without additional prompt text.
- `/quit` or EOF ends the session and writes the report. Ctrl-C at the input
  prompt also ends normally; interruption during model execution is recorded as
  a failed/interrupted execution, not a completed answer.

Multiple files can be queued. Their names and text are inserted into the user
message, then retained in conversation history. Files are not uploaded through a
provider file API, and PDF/image/binary decoding is not supported. Attached text
is sent to the configured provider in live mode and appears in local trace/report
content unless `--metadata-only` is selected.

Console requires `--live`: fixed offline answers would be misleading for arbitrary
human questions. To run the same static test against a real provider, use `--live`
with the default static client.

## Reports and checks

The command prints its HTML report path and saves `spans.jsonl`, `run.json`,
`prices.json`, and `report.html` under a fresh `reports/simple_chat/` directory.
`--out` selects a new directory. Reports are finalized when the session ends;
quitting before any request produces an incomplete report with no model spans.

In the static run, compare R1's fresh input with R2's cached history and new prompt.
Edit test prompts in `test_case.py`, not the agent file. When changing the
scenario in offline mode, update the corresponding answers in `test_case.py`.
Tests in `tests/test_simple_chat_clients.py` exercise file context, console input,
history, and the interrupt boundary without contacting a provider.

All samples follow this structure; Langfuse variants share the same workflows.
