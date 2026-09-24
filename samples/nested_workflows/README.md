<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Nested workflows and context scopes

## The lesson

This sample demonstrates **workflow nesting and multiple levels of context**.
It builds on the shared-history / isolated-child ideas in `context_budget`, but
it does **not** install context budgets, summarizers, compaction triggers, or
artificially long conversation fixtures.

The coding supervisor belongs **inside the coding workflow**, sharing its
working history with the coder. It communicates with the outer planner through
explicit assignments and result packets, not a second shared conversation.

## Workflow nesting and context sharing are different things

| Context | Participants | Retention | Deliberate handoffs |
| --- | --- | --- | --- |
| Level 1: outer delivery context | Planner and tester | One evolving message history throughout delivery | Task and defects sent down; review-approved artifact or blocked result returned up |
| Level 2: coding context | Coding supervisor and coder | One evolving message history across review rounds **and tester-driven re-entry** | Current artifact and review task sent down; review verdict and findings returned up |
| Level 3: isolated review context | Reviewer | **Fresh history for every review invocation** | Requirements, current candidate, and reported defects in; validated assessment out |

“Shared” means agents receive the same evolving conversation, not that their
system instructions are merged or that a shared model object holds their memory.
Each role gets its own instructions on each model call. In the demo, model
adapters also have independent response queues.

The host retains coding history in `DeliveryState.coding_memory`, but neither
planner nor tester receives that field in a model request. Only `messages` and
an explicit packet are passed to an outer role. That makes this **model-context
isolation**, not a process, credential, or security sandbox. The host and trace
recorder can inspect all scopes.

## Ownership

Agents and their response contracts live in `src/agent_runtime/agents/` and can
be reused independently. `workflows/nested_workflows.py` owns this sample's
outer, coding, and review graphs; `workflows/nested_policy.py` owns its loop
guards. No workflow from another sample is reused. The three catalog entries
are scenarios of this one sample.

## Actual execution structure

The outer `StateGraph` invokes a compiled coding `StateGraph` from its
`coding_workflow` node. The coding graph invokes a compiled one-node review
graph from its `isolated_review` node. These are actual nested calls, not a
flattened graph with decorative hierarchy labels.

The outer sequence is:

`planner → coding workflow → planner → tester → planner`

The coding sequence is:

`coding supervisor → coder → isolated review → coding supervisor`

A failed review sends control to the coder through the coding supervisor. A
passed review returns the artifact to the planner for testing. A tester defect
returns through the planner into the **same retained coding context**. Passing
both gates completes the task; either exhausted guard produces a blocked result.

The reviewer is a scoped child invocation. The coder is a peer of the supervisor
within their shared coding context. Using the word “subagent” is not what creates
isolation: the deliberately selected input history and handoff fields do.

## Default story: both loops are visible

The agents work on `normalize_tags(tags)`: trim and lowercase strings, remove
duplicates while preserving order, drop blanks, and avoid mutating the input.

| Step | Event | Context behavior |
| --- | --- | --- |
| 1 | Planner dispatches the task | Only the structured task enters coding context |
| 2 | Supervisor asks coder for candidate 1 | Both share coding history |
| 3 | Reviewer rejects duplicate handling | Review context 1 receives only a review packet |
| 4 | Supervisor asks coder for candidate 2 | Prior code and returned findings remain visible |
| 5 | Reviewer passes candidate 2 | Review context 2 starts fresh |
| 6 | Coding workflow returns; planner dispatches tester | Only the explicit result enters outer history |
| 7 | Tester finds blank-tag handling defect | Planner and tester share this result |
| 8 | Planner re-enters coding with defects | The previous supervisor/coder history survives |
| 9 | Coder produces candidate 3 | It sees earlier attempts and returned test defects |
| 10 | Reviewer passes candidate 3 | Review context 3 starts fresh |
| 11 | Planner dispatches tester, tester passes, planner finishes | Outer shared history contains both testing rounds |

The default script intentionally models a reviewer overlooking one acceptance
issue in candidate 2. This is an authored demonstration, not evidence of a real
model's review quality. Its queues contain **18 agent calls**: 5 planner,
5 coding supervisor, 3 coder, 3 reviewer, and 2 tester. No summary-model calls
are present. The saved [HTML report](../../reports/nested_workflows/report.html) and
[Excel report](../../reports/nested_workflows/report.xlsx) contain the latest live-model execution through LangGraph. Its call counts and
repair paths can differ from the deterministic demo story above.

## Context lifecycle and handoff rules

The outer planner creates a stable task contract. Subsequent planner calls must
preserve it; test-driven repairs cannot silently weaken acceptance criteria.

On coding re-entry, the runtime supplies the previous coding history and total
attempt count, but resets the visit's review decision and local round count.
This prevents a previous review approval from skipping development after a
new tester failure. The outer coding-cycle count is not reset.

Each coder result separates `source` from `working_notes`. The review packet
contains the task, current artifact, and reported defects. It excludes coder
notes, supervisor directives, the outer transcript, and earlier review
transcripts. The returned assessment is deliberately shared with the coding
supervisor; “isolated” does not mean that review findings cannot be returned.

Every assessment must identify the current candidate. A stale candidate ID,
contradictory approval, missing failure findings, malformed JSON, or a request
to skip a required gate raises an error instead of taking a success edge.

Nested graphs use `checkpointer=False`. Their histories are supplied explicitly;
there is no hidden child session accumulating additional context. The outer
harness may still retain outer state. Durable pausing/resuming inside a child
is outside this lesson.

## Circuit breakers

The sample implements deterministic finite-loop guards rather than a timed
open/half-open retry circuit breaker. Agents receive the permitted next action;
the runtime validates that their dispatch respects it.

| Guard | Default | Counts | Exhaustion behavior |
| --- | --- | --- | --- |
| Inner review guard | 3 | Coder/reviewer rounds within one coding visit, including the first | Return `blocked`; the tester is not invoked |
| Outer test-fix guard | 3 | Coding-workflow visits for this delivery task, including the first | Return `blocked` with unresolved tester findings |

Approval on the last allowed attempt is still accepted. A failed last attempt
is not relabeled as success. Defaults bound coder attempts to at most nine per
task. A blocked child cannot be automatically retried by the outer workflow.
There are no unbounded retries for malformed output. These guards limit loops;
they are not provider-request timeouts or a full operational reliability policy.

## Run from the lg-report repository root

`sample.json` registers the sample and both failure variants through the
existing dynamic catalog.

```sh
uv run python -m agent_runtime --sample nested_workflows --prices models.json --demo
uv run python -m agent_runtime --sample nested_workflows_review_limit --prices models.json --demo
uv run python -m agent_runtime --sample nested_workflows_test_limit --prices models.json --demo
```

The expected report location for the default sample is
`reports/nested_workflows/report.html`, following the existing runner's sample
output convention. The saved report uses real models with scripted user input. Use `--demo` to reproduce both authored repair loops.

`scenario` selects offline fixture queues only. It does not force a verdict in
live mode. The workflow factory also accepts `max_review_rounds` and
`max_coding_cycles`. Substantially larger values may require a larger LangGraph
`recursion_limit`; that framework guard is a backup, not the business policy.

## State diagram and recorded collaboration

The HTML report places a workflow state diagram first, before the cost chart.
The state diagram shows the actual declared nodes and decision edges, including
branches that were not taken. Nested boxes show delivery containing coding, and
coding containing isolated review. Dashed links expand child workflow calls.
The shared execution harness extracts its definition automatically from the
compiled graphs and saves it with the trace. The sample exposes references to
the checkpoint-disabled child graphs behind its state-projection wrappers;
it does not maintain a separate diagram's nodes or edges. Rerendering needs
neither an LLM nor current source code. The collaboration diagram shows
recorded agent activity.

Agent names are their roles: `planner`, `coding_supervisor`, `coder`, `reviewer`,
and `tester`. The graphs are nested; a naming prefix does not establish nesting.

## What to inspect in the trace

Open the nested coding workflow beneath the outer dispatch and the reviewer
invocations beneath it. Model-call metadata includes `report_history_id`:

- `outer:<task-id>` is shared by planner and tester.
- `coding:<task-id>` is shared by supervisor and coder, unchanged on re-entry.
- `review:<task-id>:<attempt>` differs for every reviewer call.

`report_context_depth` is saved as metadata (1, 2, or 3). Explicit history
labels distinguish the outer, coding, and individual review contexts in the
cost chart. The state diagram shows graph nesting, while callback ancestry,
context identifiers, and captured prompts provide execution evidence.

Small sentinel strings make leakage tests readable: `OUTER_ONLY_DETAIL` stays
in outer history; `CODING_ONLY_DETAIL` stays in coding history;
`REVIEW_ONLY_DETAIL` occurs only in the reviewer's role instructions.

## Teaching fixtures versus real development

This lesson has no filesystem, shell, or test-runner tools. The coder returns
candidate source as data. In demo mode, review and test outcomes are scripted.
In live mode, the tester is a model-based acceptance assessment, **not executed
tests**. A real development harness would attach authorized workspace and test
execution tools separately; that is deliberately outside the nesting lesson.

The demo uses the existing metering helper and resets only its per-request
accounting ledger. It never resets, trims, or summarizes a shared conversation.
Usage is explicitly simulated and assumes no cache reuse.

## Verification

```sh
uv run pytest tests/test_nested_workflow_policy.py tests/test_nested_workflows_integration.py -q
```

The policy suite checks both finite-loop guards, pass-on-last-attempt behavior,
invalid contracts, stale candidate rejection, exclusion of private coder notes,
complete fixture queue consumption, and the defects in the trusted source
fixtures. The integration suite inspects actual model inputs to check isolation,
shared history on re-entry, trace identifiers, sync/async execution, and both
failure paths.

All 31 policy/fixture tests and all 8 supplied LangGraph integration tests pass
in the project environment. Three additional CLI tests verify all scenarios and
report context labels. The CLI also generates HTML and Excel from the
scripted run. The saved report has been refreshed using real models with scripted user input.
Its model decisions and provider usage are live; the deterministic tests above
use scripted models to exercise the specified repair and failure paths.

## Sources used for the adaptation

Repository snapshot: `martinbechard/lg-report` at
`6defd618d8ca26c7829aab03c365b0eba5c8fca9`.

Relevant existing modules: `workflows/context_budget.py`,
`workflows/review_loop.py`, `agents/evidence_judge.py`,
`harness/sample_catalog.py`, and `harness/simulated_model.py`.

LangGraph's [subgraph documentation](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
describes the node-wrapper pattern for mapping different parent/child state
schemas and the explicit choice of per-invocation or stateless child persistence.
