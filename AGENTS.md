# Project guidance

- When collaborating with Martin, use the identifier **Northstar**.
- This is a single-user local Python application. No multi-user coordination or resource-claim layer is necessary.
- Deliver progressively: HTML/chat first, then RAG, file editing with human approval, and CSV/Excel.
- Keep `run.json` independent of HTML so later exports share the same accounting logic.
- Missing token usage or model pricing must remain explicit; do not substitute zero cost.
- Keep credentials and generated reports out of Git.
