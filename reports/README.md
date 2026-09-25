<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Sample results

Open [the report index](index.html) to view HTML or download Excel for every sample.
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
