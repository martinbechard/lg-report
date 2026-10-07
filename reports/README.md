<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Sample results

Open [the report index](index.html) to view HTML or download Excel for every sample.
The [model comparison](model_comparison/report.html) combines separate live
GPT-5.5 and GPT-5.6 Luna runs through both OpenAI and Codex, plus Codex GPT-6 Sol, all starting from the same simple-chat request. GPT-6 Luna (Codex, high effort)
plays the user in every run, adapting follow-ups to each assistant. Conversations
may therefore end at different turns. The shared goal covers workflow steps and
tool observations, with no minimum turn count. Each run records the user model's
stop explanation and distinguishes goal completion from the safety cap.
The comparison shows only the tested assistants’ costs and timings.
Its QA assessments share the saved `model_comparison/qa-rubric.json`: binary
goal checks, anchored quality checks, and common speed/cost formulas. The
report displays these criteria once above the comparison table.
Codex uses its local CLI and account; its usage includes harness overhead.
Its estimated cost uses the corresponding OpenAI API token rates, with the
exact mapping retained in the saved pricing aliases. Its nested
model folders retain `run.json`, `prices.json`, and `spans.jsonl`. These are single
observations with captured answers, not quality rankings or latency benchmarks.
Regenerate the HTML with the `lg-report compare` command documented in the root README.
No model credentials, Python installation, or sample execution are needed to inspect
these saved outputs. The index identifies completed and failed runs from the latest
batch. The saved reports use real provider models with scripted user requests
and approval/clarification answers, except the two nested-workflow reports,
which are explicitly simulated examples of parent dispatch and child review.
Their live runs may take different review paths. The obsolete tester-loop
variant was removed when the nested sample was simplified. In the circuit-breaker lesson, normal workflow termination means the
breaker stopped the repeated failure, not that the file was written.
The saved quote example completed after the live model asked about quantity and
addressing. Its answer uses the authored fixture: 500 households, identical
invitations, and 500 addressed envelopes using the supplied spreadsheet.
`run.json` records the
provider, model, execution status, and whether a run is simulated.

The state diagram before the cost chart is generated automatically from the
compiled LangGraph nodes, edges, branch maps, and discoverable child graphs.
The execution harness saves that topology with the trace; rendering needs no
LLM call or current workflow source. Opaque wrappers can expose actual child
graph references without maintaining a second list of states or transitions.
Unmapped dynamic destinations are identified as unavailable. The generator
reads DeepAgents' compiled task registry to capture available delegates.
Two-headed task links show model-selected delegation and return; they are not
unconditional workflow transitions. Identical child definitions share one box
within their parent, with a link from each call step. The collaboration view
shows which delegates actually ran.

Each stable sample folder contains its HTML and Excel reports, normalized
`run.json`, raw `spans.jsonl`, and the exact model-price and exchange-rate snapshot
in `prices.json`. Excel workbooks use the portable Python exporter installed with `uv sync --locked`.
Workbook input is saved as `excel-data.json`; applicable lessons
also include context evidence or the edited demonstration file. Reruns replace
the previous output in the same folder.

```sh
uv run scripts/run_samples.py              # Real models; quote questions use the terminal
uv run scripts/run_samples.py --simulated  # Explicit offline models
```

The live batch first runs `scripts/update_exchange_rate.py` once to refresh the
shared `exchange-rate.json`. Individual samples only read this saved reference;
a failed refresh preserves the previous rate. Missing/invalid saved FX makes EUR
unavailable without preventing model execution. `--fx-file PATH` uses another
saved reference and skips the refresh. Simulated batches also skip the refresh.

Generated logs, ad hoc output folders, and Langfuse state remain ignored.
PNG workbook previews are not generated; inspect the HTML or Excel report directly. The root [LICENSE](../LICENSE) and
[NOTICE](../NOTICE) cover the report files; third-party excerpts retain their
source terms.

The claims comparison now uses `edit-with-patched-state` (formerly
`claims_context_naive`) and `edit-with-reloaded-state` (formerly
`claims_context_managed`). Saved report folders and display titles use the new
names. Raw traces and context audits retain the original execution, including
historical mode labels; this rename did not rerun or simulate those live calls.

## QA evaluation example

`qa_evaluation/` records a live `simple_chat` execution through Codex GPT-6 Luna,
with the authored fixed user prompts and one independent Codex GPT-6 Sol judge.
The October 6, 2026 recording contains all four QA scores, reasons, overall score,
and separate judge receipts in `run.json`. HTML and Excel display the saved
assessment. Scores are subjective rubric judgments, not benchmark results.

Reproduce with `LG_PROVIDER=codex LG_MODEL=gpt-6-luna uv run python -m agent_runtime
--sample simple_chat --demo --live --client static --qa --prices models.json
--out reports/qa_evaluation` (as one shell command). This makes real model calls
and replaces that example; the regular batch keeps the saved example linked.

All five columns in the saved comparison were rerun on October 6, 2026 after
switching simple chat to a tool-free LangGraph agent. Both API and Codex requests
omit the eight unused application tools. Codex uses the minimal catalog profile
with no native tools, skills, or identity guidance. Every Agent uses medium
reasoning effort; the Codex GPT-6 Luna user and Sol 6 judge use high effort.
All five assessments use the same saved scoring criteria. Goal and quality
scores are calculated from the judge's criterion checks; speed and cost use
the shared numeric scales. The comparison shows turn time, time to first token, output rate,
and a cost-calculation recap after the graph. Judge cost is omitted, and
Agent performance measurements exclude both the judge and simulated user.
Both direct API runs were refreshed with streaming first-output capture.
Timing rows are shared across providers; the root README documents
client versus internal timing boundaries. QA cost assessments cover Agent
model calls only. The exchange rate is shown once when shared. Extra backend
diagnostic metrics are omitted from the report and collector.
The Codex payload investigation and remaining differences from the direct SDK
are documented in [Codex models](../docs/codex-models.md#request-payload-inspection).
