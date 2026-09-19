<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# File editing with human approval

Read a UTF-8 source and propose two successive modifications to an output file.
Each modification appends one teaching-scenario sentence. The graph performs real
file I/O; it uses no model, provider key, or simulated token counts.

```sh
uv run python -m samples.file_approval.app --source samples/file_approval/input.txt --target reports/edited-summary.txt --mode always-ask
uv run python -m samples.file_approval.app --source samples/file_approval/input.txt --target reports/automatic-summary.txt --mode autoapprove
```

The target's parent directory must already exist. `reports/` exists in this
repository and keeps the resulting local files out of Git. Paths are relative to
the working directory; absolute paths also work. Source and target may be the
same file. An existing target is replaced by the proposed source-derived content,
so the approval prompt shows its full diff and replacement text.

- `always-ask` (default): review each exact proposed write, then type `approve`,
  `reject`, or `cancel`. Unknown answers ask again without writing.
- `autoapprove`: apply both modifications without prompting.
- `reject`: skip only this modification. A later proposal excludes rejected text.
- `cancel`, EOF, or Ctrl-C at a prompt: stop remaining work. Previously approved
  changes remain; cancellation does not undo a write already authorized.

Approval applies to one modification only. Reads require no approval. If the
output changes while the prompt is waiting, the workflow fails without overwriting
that newer content. This is a single-user example, not a concurrent filesystem
transaction service.

For an unattended fixture, explicitly select simulated human decisions:

```sh
uv run python -m samples.file_approval.app --source samples/file_approval/input.txt --target reports/scripted-summary.txt --client static --decision approve
```

`--decision reject` and `--decision cancel` exercise other outcomes. Static
approval is a test fixture, not evidence that a human approved the changes.

The workflow separates read, prepare, approve, and apply nodes. Only the apply
node writes. LangGraph's [interrupt/resume API](https://reference.langchain.com/python/langgraph/types/interrupt)
suspends the approval node and resumes with the user's decision. The shared
`HumanLoop` client preserves recorder callbacks across every resume. An in-memory
checkpointer supports pauses within this process, not recovery after restarting it.

Each run prints its final state and a local `report.html` path. `run.json` remains
independent of HTML. `--out` chooses a new report directory; `--metadata-only`
omits captured payloads. `--prices models.json --fx-file /path/to/fx.json` avoids
price/FX network lookups. No LLM is used; `--live` is rejected.

Local reports default to `report.html`, `run.json`, `spans.jsonl`, and
`prices.json` in the current working directory. The next default run replaces
these files. Use `--out reports/saved-run` with a new directory to keep a run.
