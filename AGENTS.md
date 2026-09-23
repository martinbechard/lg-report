<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Project guidance

- When collaborating with Martin, use the identifier **Northstar**.
- This is a single-user local Python application. No multi-user coordination or resource-claim layer is necessary.
- Deliver progressively: HTML/chat first, then RAG, file editing with human approval, and CSV/Excel.
- Keep `run.json` independent of HTML so later exports share the same accounting logic.
- Missing token usage or model pricing must remain explicit; do not substitute zero cost.
- Keep credentials, ad hoc local reports, and Langfuse databases/reports out of Git. Check in the stable sample outputs under `reports/<sample>/` and their index so learners can inspect HTML, Excel, and supporting evidence without running models.
- Start every code file with a plain-language header explaining its purpose, responsibility, and important assumptions or relationships. Explain obscure names such as token-metering helpers; do not merely restate the filename. Keep headers accurate when moving or changing code.
- Comment code heavily to explain **WHY** it exists and **WHAT** it is trying to achieve. Document functions, classes, workflow nodes, state fields, and meaningful control-flow decisions with their intent, rationale, assumptions, side effects, and important limitations. Explain approval boundaries, retries, failure handling, and accounting rules where they occur. Ground explanations in the actual implementation; avoid line-by-line paraphrases or claims the code does not guarantee. Keep comments and docstrings current whenever code changes, including tests, scripts, and samples.
- Include `Copyright (c) 2026 Martin.Bechard@DevConsult.ca` in every code file's header, preserving any third-party copyright and license notices. Preserve accurate AI attribution alongside the purpose and rationale.
