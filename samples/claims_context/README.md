<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Claims and policy: agent-driven retrieval and context management

The **agent chooses when to read** fictional claim CLM-001 or policy POL-001
based on the human's query. Neither record is preloaded. The application never
calls a read tool on the agent's behalf, inserts an invented read request, or
forces a reload after an edit.

The claim contains `claim_id`, `revision`, `description`, and `status`. The
read-only policy contains `policy_id`, `covered_event`, `deductible_cad`, and
`limit_cad`. Both are dummy in-memory records, reset on each run. No real claim,
payment, or local source file is modified.

## Workflow

The agent encapsulates **how to do claims work**. The workflow is the harness
that controls **which context is carried into the next turn**. The same agent
works under either strategy; it receives no `naive` or `managed` setting.

```text
Variant sample.py: model configuration, human/static client, reporting
   |
   v
build_workflow(): compose a ClaimsAgent with a conversation harness
   |
   +-- Harness owns: turn loop, working history, context mode, audit
   |      |
   |      +-- Agent interface: invoke / context_version /
   |                          retain_unaffected_context / evidence
   |             |
   |             +-- ClaimsAgent owns: system prompt, tool registration,
   |                 model/tool loop, record store, dependency knowledge
   |                       |
   |                       +-- read_claim / read_policy / edit_claim
   |
   +-- Report history stays separate from model working context
```

```text
  Human query + existing working context
                    |
                    v
           +-----------------+ <---------------------------+
           | claims_agent    |                             |
           | decides action  |                             |
           +-----------------+                             |
                    |                                      |
              Tool needed? -- yes --> Agent chooses:       |
                    |                 read_claim            |
                    |                 read_policy           |
                    |                 edit_claim            |
                    |                      |                |
                    |                Tool executes          |
                    |                      |                |
                    |                Actual result ---------+
                    no
                    |
                    v
                Final answer
                    |
                    v
       Workflow detects successful edit?
              /                \
            No                  Yes
            |                    |
       Keep context         Context mode?
            |                /          \
            |             naive        managed
            |               |             |
            |          Keep history   Remove claim exchanges
            |               |         and conversation prose;
            |               |         retain actual policy reads
            +---------------+-------------+
                            |
                     Next human query
                            |
                     Agent decides again
                     (NO automatic read)
```

Each model call receives tool schemas, but those schemas do not contain the
record data. Tool execution happens only after a model-emitted tool call. In
managed mode, the workflow compares the agent's opaque `context_version` before
and after a completed turn. A change causes the harness to request and apply the
agent's `retain_unaffected_context` projection. The agent knows that claim edits
leave policy observations valid; the harness knows neither tool names nor the
claim schema. Multiple field replacements or multiple
successful edits in one turn are reflected in the store before invalidation.
Failed edits alone do not invalidate anything.

## Run the comparison

```bash
uv run python -m agent_runtime --sample claims_context_naive --show-context --out reports/claims-naive --demo
uv run python -m agent_runtime --sample claims_context_managed --show-context --out reports/claims-managed --demo
```

The demonstration asks five questions:

1. Policy deductible: the agent requests `read_policy`.
2. Claim description and status: the agent requests `read_claim`.
3. Correction from theft to accidental damage at home, status approved: the
   agent requests `edit_claim`.
4. Policy limit: the agent uses retained policy context; no claim read occurs.
5. Current claim details: in managed mode, the agent requests `read_claim` again.

**Offline tool choices and answers are scripted model responses.** The tools
really run. Both modes give the same correct scripted answer; the difference is
what context the model receives and whether it requests another read. A canned
wrong answer would not demonstrate a real model becoming confused. Use live mode
for that experiment. The application does not
select reads based on question text or mode. Mode controls context invalidation;
only the offline model fixture scripts different follow-up choices. Simulated
usage counts the actual inputs; neither mode models provider cache reuse.

Output directories are reused. Omit `--out` to replace the report bundle and
`context.json` in `reports/claims_context_naive/` or `reports/claims_context_managed/`. Use `--prices models.json` and
`--fx-file PATH` for supplied pricing and exchange snapshots. FX otherwise reads the shared
`exchange-rate.json` without a lookup. Use `--demo` to keep model calls scripted even when an API key is configured.

## Compare actual agent decisions

```bash
cp samples/claims_context/.env.example samples/claims_context_naive/.env
cp samples/claims_context/.env.example samples/claims_context_managed/.env
# Configure provider credentials and model in each variant file.
uv run python -m agent_runtime --sample claims_context_naive --live --mode naive --show-context --out reports/claims-live-naive
uv run python -m agent_runtime --sample claims_context_managed --live --mode managed --show-context --out reports/claims-live-managed
```

These runs share user questions, tools, and agent instructions. The real model
chooses which record to load, when to edit, whether to reread, and how to answer.
For your own queries, replace `--demo` with `--client console --live`; `/quit` ends the session.
Console `/attach` supplies ordinary text context, not a stored claim or policy.

A live model may answer correctly in both modes, reread proactively, or make a
mistake after invalidation. Inspect actual tool choices and results; the sample
has not established a live accuracy improvement. In naive mode, current claim
facts must be reconstructed from the last read plus **all successful subsequent
replacements**, in execution order. An edit acknowledgement is not a full claim.

## What the managed mode removes

At edit-turn end, the harness elects to apply the agent's retention rules.
Those rules remove claim tool exchanges and all prior
conversation prose, including user corrections and assistant paraphrases that
could repeat stale facts. It preserves genuine `read_policy` call/result pairs
with their original call IDs. For mixed batches, it keeps only the policy calls
and corresponding results. It never adds a new tool call.

An application notice says that claim context was invalidated. It supplies no
claim contents and makes no retrieval decision. A policy-only question can be
answered without loading the claim; a claim-dependent question leaves it to the
agent to request a fresh read.

This is a deliberately coarse single-claim teaching policy. It loses historical
discussion and user preferences expressed in prose. It does not purge between
tool calls in the same turn. A larger application would track dependencies and
preserve unrelated conversational constraints separately. The minimal LangChain/
LangGraph tool loop avoids DeepAgents automatic summarization so this policy is
visible in the captured messages.

## Inspect the evidence

- `--show-context` prints every actual model input.
- `context.json` saves inputs, message counts, invalidation events, and the final
  authoritative claim. This audit is never fed back to the model.
- `report.html`, `run.json`, and `spans.jsonl` retain the original execution and
  accounting evidence, including messages removed from working context.
- `--metadata-only` omits content from saved audits/reports and suppresses
  `--show-context`. Terminal user prompts and answers remain visible.

Examples: [naive report](../../reports/claims_context_naive/report.html)
and [managed report](../../reports/claims_context_managed/report.html).
The batch runner includes both modes.

Files and responsibilities follow the project's agent/workflow separation:

- `src/agent_runtime/agents/claims_agent.py`: encapsulated agent with system prompt,
  tool registration, model/tool loop, private domain state, and knowledge of
  which context remains valid. The model chooses its actions here.
- `src/agent_runtime/tools/claims.py`: dummy records and read/edit tool implementations.
- `src/agent_runtime/workflows/claims_context.py`: participant composition and the
  harness controlling turns, naive/managed strategy, and independent audit.
  It never reads records or inspects tool names to decide what the agent does.
- `sample.py`: scripted user prompts and model responses.
- [Naive metadata](../claims_context_naive/sample.py) and [managed metadata](../claims_context_managed/sample.py): one context-policy variant per file, each sharing the `claims_context` implementation.

Tests verify on-demand reads, policy retention, no forced reload, multiple edits,
failed edits, tool-pair integrity, and unchanged audit evidence.

## Angular client

Select **Claims context — naive** or **Claims context — managed** in the catalog,
or add `--client angular --mode managed` when launching this sample. Both clients
use LangGraphAgent and the same context-policy graph. Display history is separate
from working model context; retaining a message in the UI does not reintroduce it
to the model. **View context audit** exposes the separate audit. Content follows
the shared launcher's capture setting: actual model inputs are included by
default; `--metadata-only` retains counts without captured content.
