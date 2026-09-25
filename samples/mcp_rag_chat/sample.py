"""Author the mcp rag chat example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. The shared catalog
constructs the offline model from CONVERSATION; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "mcp_rag_chat",
    "name": "MCP Wikipedia RAG",
    "description": "Reach the local Wikipedia index through MCP.",
    "mcp_tools": ["semantic_search_wikipedia"],
}

CONVERSATION = [
    {
        "role": "client",
        "content": "How has the Australian raven adapted to urban environments?",
    },
    {
        "role": "wikipedia_mcp_agent",
        "content": "",
        "tool_calls": [
            {
                "name": "semantic_search_wikipedia",
                "args": {
                    "query": "How has the Australian raven adapted to urban environments?"
                },
                "id": "wiki-mcp-search-1",
            }
        ],
    },
    {
        "role": "tool",
        "name": "semantic_search_wikipedia",
        "tool_call_id": "wiki-mcp-search-1",
        "content": "Expected: actual semantic_search_wikipedia result for {'query': 'How has the Australian raven "
        "adapted to urban environments?'}; supplied by the running tool.",
    },
    {
        "role": "wikipedia_mcp_agent",
        "content": "The MCP search completed. Inspect the semantic_search_wikipedia result for retrieved passages "
        "and source IDs. This scripted response demonstrates retrieval and accounting, not answer "
        "synthesis.",
    },
]
