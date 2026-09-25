<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Nested workflows

This sample teaches one idea: **a parent dispatches work to a child that owns
its review loop**.

```text
Parent:  planner → review workflow → finish
                       │     ↑
Child:            author → judge
                     ↑       │
                     └ revise┘
                         approve or limit → return
```

The planner turns the user request into one assignment. The parent invokes the
compiled child graph with that assignment. The child author writes a draft,
the judge reviews it, and a rejection sends feedback back to the author.
Approval or the round limit returns the result to the parent. The parent
publishes it unchanged, including unresolved feedback when review did not pass.

The author and judge are the same reusable agents used by
[`review_loop`](../review_loop/README.md). This sample defines its own child
workflow rather than importing another sample's workflow. Read
[`nested_workflows.py`](../../src/agent_runtime/workflows/nested_workflows.py)
for both levels, and [`work_planner.py`](../../src/agent_runtime/agents/work_planner.py)
for the parent's role instructions. Each graph uses ordinary named nodes and
explicit edges, just like `review_loop`.

## Scripted example

[`sample.py`](sample.py) keeps the complete exchange in `CONVERSATION`:

1. The user requests a tag-normalizing function.
2. The planner prepares an assignment.
3. The author writes a draft that leaves duplicates and blanks.
4. The judge requests those fixes.
5. The author revises the draft.
6. The judge approves it and the result returns to the parent.

That is five model calls: one planner call and two author/judge rounds.
The [`review limit` variant](../nested_workflows_review_limit/README.md) allows
only one round, so the first rejection returns an explicitly unapproved draft.

```bash
uv run python -m agent_runtime --sample nested_workflows --demo
uv run python -m agent_runtime --sample nested_workflows_review_limit --demo
```

Replace `--demo` with `--live` to use configured provider models. In live mode,
the number of review rounds depends on actual responses. A review is a model
assessment: neither mode executes the generated function or runs its tests.
The default maximum is three drafts, including the first. Approval on the last
allowed round succeeds; a rejection on that round returns unresolved feedback.
Invalid judge output fails visibly.

Saved [HTML](../../reports/nested_workflows/report.html) and
[Excel](../../reports/nested_workflows/report.xlsx) reports show the explicitly
simulated example. The state diagram captures the actual parent and child
graphs; the activity records show which path executed.

Run focused verification with:

```bash
uv run pytest tests/test_nested_workflows_integration.py tests/test_review_loop.py -q
```
