<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Simple chat — an agent with interchangeable clients

## Purpose

Learn to separate an agent's behavior from its user and its test cases. The agent
receives messages at runtime; it does not know the test prompts, read terminal
input, or generate reports. The session retains history so later requests include
earlier user messages, attachments, assistant answers, and any tool exchanges.

## Files and responsibilities

The reusable code lives under `src/agent_runtime/`:

- **`workflows/simple_chat.py`** composes the single-agent workflow.
- **`agents/chat_agent.py`** defines the named agent, its system instructions,
  and its DeepAgents graph. It imports no client or test case.
- **`harness/conversation.py`** defines requests, text attachments, the client
  interface, and the session loop that retains history and turn metadata.
- **`harness/console_client.py`** accepts human prompts and text files.
- **`harness/script_prompter.py`** sequences authored requests for `ConsoleClient`;
  it contains no predefined prompts or expected answers.
- **`harness/simulated_model.py`** and **`harness/demo_meter.py`** provide the
  generic offline model and context/token simulation.
- **`src/reporting/`** normalizes captured traces, prices usage, and exports HTML/Excel.
  Runtime trace capture lives in `src/agent_runtime/harness/trace_capture.py`.
- **`tools/`** is the home for application tools. Simple chat defines none.

This sample directory contains **`sample.py`** with `SAMPLE` discovery metadata,
the scenario's user prompts, prerecorded model answers, and model factories,
plus this README and configuration example. A client simulator and a model
simulator serve different roles even when one test case configures both.

A future user-avatar agent can implement `receive()` and `respond(result)` from
`ChatClient`: after receiving an answer, it chooses the next request, or returns
`None` to end. Such an avatar would live in its own named file under `agents/`.
No avatar agent is implemented yet.

## Run the scripted test case

```bash
uv sync
uv run python -m agent_runtime --sample simple_chat --demo
```

`--demo` selects fixed prompts and a simulated model. With a provider key configured, omitting `--demo` starts live console chat. This executes a real graph
with two predefined user turns and two prerecorded responses, without provider
charges. Model-price refreshes may still access the network; use `--prices models.json`
to avoid them. FX always reads the saved shared `exchange-rate.json`.

The graph includes DeepAgents' built-in tool definitions, which consume
context even though its instructions ask it to answer without tools. StateBackend
keeps built-in file operations in graph state rather than the local filesystem.

## Console chat

```bash
cp samples/simple_chat/.env.example samples/simple_chat/.env
# Configure an OpenAI or Anthropic API key in that file.
uv run python -m agent_runtime --sample simple_chat --client console --live
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

Console requires live mode (selected automatically when a provider key is configured): fixed offline answers would be misleading for arbitrary
human questions. To run the same scripted test against a real provider, replace `--demo` with `--live --client static`.

## Reports and checks

The command prints its HTML report path and saves `spans.jsonl`, `run.json`,
`prices.json`, and `report.html` in `reports/simple_chat/` (replacing the previous default run).
`--out` selects another reusable directory. Reports are finalized when the session ends;
quitting before any request produces an incomplete report with no model spans.

In the scripted run, compare R1's fresh input with R2's cached history and new prompt.
Edit test prompts in `sample.py`, not the agent file. When changing the
scenario in offline mode, update the corresponding answers in `sample.py`.
Tests in `tests/test_simple_chat_clients.py` exercise file context, console input,
history, and the interrupt boundary without contacting a provider.

All samples follow this structure; Langfuse variants share the same workflows.

## Why the factory is a function

Importing `sample.py` makes its metadata and conversation available. It does not
create a model. In simulated mode, `build_scripted_models(options)` calls
`make_simulated_model()` to create a fresh response cursor and usage ledger for
the conversation. Live mode bypasses these factories and selects the configured
provider. The factory is the same one previously used by this lesson.
