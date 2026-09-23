<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Sample results

Open [the report index](index.html) to view HTML or download Excel for every sample.
No model credentials, Python installation, or sample execution are needed to inspect
these saved outputs. The existing collection contains real `gpt-5.6-luna` calls. The new
`shell_script` report uses a simulated model and real shell execution.
`context_budget` uses simulated agents and summaries with real middleware
compaction, shared peer history, and isolated delegation;
`run.json` records the execution status and whether a run is simulated.

Each stable sample folder contains its HTML and Excel reports, normalized
`run.json`, raw `spans.jsonl`, and the exact model-price and exchange-rate snapshot
in `prices.json`. Workbook input is saved as `excel-data.json`; applicable lessons
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

Generated logs, workbook previews/inspection files, ad hoc output folders, and
Langfuse state remain ignored. The root [LICENSE](../LICENSE) and
[NOTICE](../NOTICE) cover the report files; third-party excerpts retain their
source terms.
