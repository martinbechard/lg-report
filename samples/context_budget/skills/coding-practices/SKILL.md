# Coding practices for a small Python change

This guidance is part of the context-budget sample. It is deliberately self-contained: no external skill, service, repository-specific command, or server is required. Apply it to the local exercise described in the plan. The plan is the durable coordination record; conversation text is transient.

## Establish the task

Read the current plan before choosing work. Identify the requested behavior, the intended user, inputs and outputs, file ownership, and acceptance criteria. Separate requirements from implementation guesses. If a task cannot be completed because information is missing, record the exact gap in the plan rather than silently inventing a contract. Prefer a small explicit contract over speculative flexibility. Make names describe responsibilities and avoid adding an abstraction that has only one use. If prior tasks changed the plan, use its latest on-disk version, not an older summary of it.

## Design before editing

For each task, describe the observable before and after behavior. List one or two representative examples and the most important edge cases. Decide where input validation belongs and what should happen for invalid values. Choose deterministic behavior that can be checked without external services. Keep the public function small and place supporting logic near it. Avoid hidden state and avoid catching errors so broadly that a failed edit looks successful. Do not make a safety claim that the code cannot enforce.

## Implement in small verified steps

Read the existing file before editing it. Change only the files named by the plan. Preserve content outside the task. When creating code, include a plain-language file header stating purpose, responsibilities, assumptions, AI assistance, and copyright. Explain non-obvious control flow and why a particular branch exists. Comments should clarify intent, not repeat syntax. After each tool result, check whether the operation succeeded before recording progress. If an edit fails or replaces the wrong string, stop and repair that step. Never record a task as complete simply because a tool was called.

## Tests and evidence

Write tests for external behavior, important boundaries, and regressions the task is meant to prevent. Do not write a test that only copies the implementation's branching. A test should distinguish a plausible bug from correct behavior. Keep test data readable. If this workflow cannot execute tests, say that they were written but not run; never substitute a code review for execution evidence. A reviewer may inspect coverage and syntax, but a review report is not a passing test result.

## Plan maintenance

The planner marks a task complete only after the independent reviewer approves that task's current files. An existing artifact alone is not approval. The worker reports concrete tool outcomes but does not edit the plan. The planner uses a narrow replacement that preserves other tasks, with concise evidence such as the created file, test name, or reviewer finding. Read the plan again after every status edit, even if the edit succeeded, so the next decision uses the current document. Keep a pending, complete, or blocked status for every task. Record uncertainty under the relevant task instead of hiding it in a final answer. If the reviewer requests changes, the same task stays incomplete. The worker repairs it and the reviewer checks the new version before the planner records completion.

## Handoff and review

The planner supplies a self-contained task assignment: the plan path, code paths, intended behavior, and acceptance criteria. The workflow forwards that assignment to the reviewer after the worker returns. The reviewer starts with an isolated context and has no access to this conversation. Ask for source-backed findings with severity and uncertainty. The workflow delivers the independent verdict. A rejection returns the same task to the worker. Approval lets the planner record completion and reread the plan before assigning another task. Report only what the visible evidence supports.

## Context limits

Large instructions, file reads, tool schemas, and tool results consume context. Use compaction to preserve durable decisions and current task state when earlier conversation is summarized. The on-disk plan remains authoritative; after compaction, reread it before the next task. Do not infer that a summary retains every line of a file. When a read is truncated, request the missing section. Keep final messages concise enough to show the task status, verification level, and next action without duplicating the entire plan.

Role ownership survives compaction: only the worker edits source/tests, only the reviewer issues approval, and only the planner updates plan status. The current workflow assignment and verdict override incomplete history summaries.
