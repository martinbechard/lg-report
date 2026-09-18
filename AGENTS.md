<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Project guidance

- When collaborating with Martin, use the identifier **Northstar**.
- This is a single-user local Python application. No multi-user coordination or resource-claim layer is necessary.
- Deliver progressively: HTML/chat first, then RAG, file editing with human approval, and CSV/Excel.
- Keep `run.json` independent of HTML so later exports share the same accounting logic.
- Missing token usage or model pricing must remain explicit; do not substitute zero cost.
- Keep credentials, local reports, and Langfuse databases/reports out of Git. Regenerated sample reports under `reports/examples/` are checked in.
- Start every code file with a plain-language header explaining its purpose, responsibility, and important assumptions or relationships. Explain obscure names such as token-metering helpers; do not merely restate the filename. Keep headers accurate when moving or changing code.
