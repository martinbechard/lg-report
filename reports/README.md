<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Sample results

Open [the report index](index.html) to view HTML or download Excel for every sample.
No model credentials, Python installation, or sample execution are needed to inspect
these saved outputs. The index identifies completed and failed runs from the latest
batch. The saved collection contains 14 successful live `gpt-5.6-luna` runs,
with provider-reported usage for all 79 model calls. `run.json` records the
provider, model, execution status, and whether a run is simulated.

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
