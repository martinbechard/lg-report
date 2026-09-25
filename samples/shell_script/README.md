<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Shell script through DeepAgent

This sample asks an agent to run `sh ./summarize.sh`. DeepAgent supplies the native
`execute` tool because `ShellBackend` implements `SandboxBackendProtocol`.
The script reads three fictional orders, writes `summary.txt`, and prints:

```text
Orders: 3
Total units: 9
```

The offline model scripts the command choice, then quotes the real tool result.
It makes two model calls and one real shell-tool call. A failed script appears
as failure in the final answer; the fixture does not invent totals.

```sh
# Offline model; real local shell execution and HTML report
uv run python -m agent_runtime --sample shell_script --prices models.json --demo

# Real model using the same backend and bundled script
uv run python -m agent_runtime --sample shell_script --live --env-file .env.local --prices models.json

# Angular client, using the same workflow
uv run python -m agent_runtime --sample shell_script --client angular --prices models.json --demo
```

The sample requires a POSIX host with `sh`, `awk`, and `cat` (macOS or Linux).
Each conversation gets a temporary working directory with `summarize.sh` and
`orders.csv`. Workspace preparation marks the copied script executable, so both
`./summarize.sh` and `sh ./summarize.sh` work. The script's output is captured in the report before that workspace
is released. Reports default to `reports/shell_script/`; `--out` selects another
folder. [Saved HTML](../../reports/shell_script/report.html) and
[Excel](../../reports/shell_script/report.xlsx) use a simulated model and a real script.

## Ownership and execution

1. `sample.py` registers the lesson. The shared launcher selects a client;
   the workflow prepares its workspace and obtains its model from `build_model`.
2. `workflows/shell_script.py` selects the shell agent.
3. `agents/shell_agent.py` creates a DeepAgent with `ShellBackend`.
4. The model requests `execute(command="sh ./summarize.sh")`.
5. `backends/shell_backend.py` runs the command and returns `ExecuteResponse`.
6. The model reads combined stdout/stderr and exit status before answering.

The backend reuses `FilesystemBackend` for native file tools. Its `id` identifies
one instance; inherited `aexecute` runs synchronous execution in a worker thread.
Commands default to a 30-second deadline; `timeout=0` disables it. A timeout kills
the command process group and returns status 124. Nonzero process statuses remain
unchanged. Returned output is limited to 20,000 bytes with an explicit truncation flag.
Earlier filesystem effects are not rolled back after failure or timeout.

Despite the protocol name, this is **local execution, not an OS sandbox**.
The temporary directory separates sample files; shell commands can access the host
with the current user's permissions and environment. This sample has no human
approval gate. Use trusted scripts and requests, including with `--live --env-file .env.local`.
