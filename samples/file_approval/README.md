<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Agent-invoked tools with automatic human approval

The human asks an agent to edit a document. The agent chooses when to read and
what to write. The workflow supplies middleware that automatically intercepts
restricted `write_file` and `edit_file` calls **after the model proposes them and before they
execute**. Approval is enforced by code, not by asking the model to behave.

```text
Human request
     |
     v
+---------------- File editor agent ----------------+
| system prompt + bounded file tools + model loop   |
|                                                  |
| model -> workflow-supplied approval middleware    |
|   ^                 |                            |
|   |          unrestricted read / approved write   |
|   |                 v                            |
|   +------------- real tools                      |
|   +------------- rejection result (no execution) |
+--------------------------------------------------+
                      |
       restricted write in always-ask mode
                      v
              checkpoint + interrupt
                      |
                human decision
                      |
              resume same checkpoint
              /         |          \
          approve     reject       cancel
          execute    agent sees    graph ends
           tool      rejection     without write

Model final answer -> END
```

The middleware is inside the compiled agent graph so it can control the actual
tool execution boundary. Ownership remains separate: `agents/file_editor.py`
constructs a DeepAgent with its native `read_file`, `write_file`, and `edit_file`
tools. `backends/file_access_backend.py` maps `/source.txt` and `/target.txt` to trusted
local paths and delegates file operations to DeepAgent’s `FilesystemBackend`.
Only the target can be changed; other backend operations are unsupported.
`workflows/file_approval.py` configures the policy;
`middleware/restricted_tool_approval.py` implements approval, cancellation, and
stale-content checks. The caller supplies the checkpointer. The client submits AG-UI resume entries to `LangGraphAgent`, preserving the thread and recorder callbacks. It does not choose when to interrupt.

The small `RestrictedToolApproval` middleware makes the policy visible for
teaching. It uses LangGraph's real `interrupt` and LangChain's agent middleware
hooks. It is application code, not DeepAgents' built-in approval middleware.
LangChain also provides `HumanInTheLoopMiddleware`; this sample spells out its
own approve/reject/cancel protocol to show cancellation ending the graph.

```sh
# Scripted model decisions; real tools and console approval
uv run python -m agent_runtime --sample file_approval --source samples/file_approval/input.txt --target reports/edited-summary.txt --demo --client console

# Real model decisions, with exactly the same automatic approval gate
uv run python -m agent_runtime --sample file_approval --live --env-file .env.local --source samples/file_approval/input.txt --target reports/edited-summary.txt --request 'Add a short next-steps section.'

# Explicitly bypass human approval for this run
uv run python -m agent_runtime --sample file_approval --source samples/file_approval/input.txt --target reports/automatic-summary.txt --mode autoapprove --demo
```

Configure `LG_PROVIDER`, `LG_MODEL`, and the provider key in the shell or
the repository `.env.local` (copy the root `.env.example` if needed). The command
above selects that file explicitly. The runtime defaults to 32,768 output tokens
so replacement documents need no sample-specific limit; `LG_MAX_TOKENS` overrides it. Custom `--request` text requires live mode.
The default offline fixture authors two successive additions and fresh reads;
it proves tool interception and execution, not live editing quality. Rejection
handling is scripted in that fixture; a live model decides how to continue from
the rejection result under its instructions.

- `always-ask` (default) shows the target path, tool name, complete proposed
  content or string replacement, and expected previous content for each restricted call.
- `approve` releases that exact call. `reject` supplies an error tool result
  without executing it, then lets the agent continue. Invalid answers ask again.
- `cancel`, EOF, or Ctrl-C at the approval prompt ends the graph. Earlier writes
  remain; there is no rollback. A batch is reviewed before any of its tools run,
  so cancellation also skips previously approved calls in that pending batch.
- `autoapprove` runs the same agent/tools without human interrupts.

The model can read only the configured source and target, and write only the
target. DeepAgent creates missing parent directories when writing. An existing target is preserved as the
starting document by the default scenario; source and target may be the same
file. Middleware checkpoints the target content when it is read (or before the
first model call if no read occurs), then compares it immediately before
executing either mutation tool. The model does not supply `expected_content`. A changed target fails the run rather than
silently overwriting a human's intervening edit. This comparison and write are
not atomic; the sample assumes a single-user local application.

For unattended demonstrations, simulate human decisions explicitly:

```sh
uv run python -m agent_runtime --sample file_approval --source samples/file_approval/input.txt --target reports/scripted-summary.txt --demo --decision approve
```

`--decision reject` and `--decision cancel` exercise other outcomes. Demo
approval is not evidence of human review. No file is read or preloaded by the
client: even the offline model builds proposals from actual tool observations.

The in-memory checkpointer supports pause/resume within this process. Resuming
replays the interrupted middleware node; it does not regenerate the model's
proposal or repeat completed tools. Restart recovery is outside this sample.

Each run prints its final state and `report.html` path. The report includes the
named file editor and real model/tool spans; offline model usage is labeled
simulated. `run.json` remains independent of HTML. `--out` selects another reusable report
directory, `--metadata-only` omits captured payloads, and
`--prices models.json --fx-file /path/to/fx.json` uses local pricing inputs.
Default output is `reports/file_approval/` and is replaced on rerun.

## Angular client

Select **File approval** in the Angular catalog, or add `--client angular` to
the CLI command with explicit source/target paths. The catalog gives each session
an isolated temporary output. Approval controls use the same driver as the
console; a download appears after a confirmed write. Starting a new chat removes
the previous temporary workspace. Explicit CLI target files are retained.
