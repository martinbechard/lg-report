<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca; third-party source excerpts retain their original rights. -->
# Application tools

`workflow_reference.py` supplies the small local workflow lookup used by the
reference-chat agent and workflow specialist. `service_evidence.py` supplies the
fictional inspection and plan checks used by the investigation agent.

Tool descriptions are model-facing contracts.
Use `@tool(parse_docstring=True)`, typed parameters, and Google-style `Args`
sections so each tool carries its own parameter descriptions into the model's
schema. Describe purpose, input meaning, and existing limits concisely; keep
agent workflow rules in the system prompt. Register these tool objects directly
with `create_deep_agent(tools=[...])`.

With content capture enabled, reports save bound definitions per model request
in `run.json` and show them in a collapsed **Tool definitions** section. Older
recordings without definitions show an explicit not-captured message.

Clients, provider configuration, conversation history, and reporting do not
belong in tools. Simple chat uses no custom evidence tools; the dispatch workflow uses
DeepAgents' native task tool for delegation.

`domain_reference.py` supplies separate movie, sports, and history lookups. These
search small source-labelled local collections by phrase, return explicit misses,
and perform no network or model calls. They are teaching retrieval tools, not a
complete vector RAG pipeline.

`search_wikipedia.py` is also published by the [Wikipedia MCP server](../mcp_servers/README.md). The server reuses this tool function; Deep Agents discovers it through LangChain’s MCP adapter.
