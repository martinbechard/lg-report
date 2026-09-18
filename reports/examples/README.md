<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Regenerated sample reports

These reports were regenerated from the current sample workflows using scripted
chat models, the checked-in `models.json` prices, and the cached September 18,
2026 USD/EUR rate. They demonstrate execution and cost estimates, not paid
provider runs. RAG and MCP RAG execute real local Chroma retrieval.

- [Simple chat](simple_chat/report.html)
- [Tool chat](tool_chat/report.html)
- [Subagent chat](subagent_chat/report.html)
- [Thinking agent](thinking_agent/report.html)
- [Review loop](review_loop/report.html)
- [Expert dispatcher](expert_dispatch/report.html)
- [RAG chat](rag_chat/report.html)
- [MCP RAG chat](mcp_rag_chat/report.html)

Each directory includes `run.json`, `prices.json`, and recorded span evidence,
keeping accounting independent of HTML. JSON and JSONL copyright notices are in
adjacent `.license` files so their machine-readable contracts remain unchanged.
Retrieved Wikipedia excerpts retain their CC-BY-SA-3.0 attribution and terms;
see [the dataset notice](../../data/rag/README.md).

Regenerate each sample into a fresh directory with:

```sh
uv run python -m samples.simple_chat.app --out reports/new-simple-chat --prices models.json
```

Review the output before replacing the checked-in example. All other local
report runs, Langfuse reports, and Langfuse database state remain ignored.
