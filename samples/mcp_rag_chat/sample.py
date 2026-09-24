"""Author the mcp rag chat example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts
from agent_runtime.harness.simulated_model import SimulatedModel

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

# User inputs and model replies are projections of the same ordered script.
USER_PROMPTS = client_prompts(CONVERSATION)


def make_simulated_model():
    """Extract AI replies into a fresh model with independent cursor and usage."""
    return SimulatedModel(
        conversation=CONVERSATION,
        metadata={
            "report_description": "Search Wikipedia through MCP and consume the returned "
            "evidence.",
            "report_effort": "light",
            "lc_versions": {"langchain-core": "1.6.3", "langchain": "1.4.1"},
        },
    )


def build_scripted_models(options):
    """Let the catalog retain this sample's report metadata in simulated mode.

    The catalog explicitly calls this callback when building a simulated run.
    options merges sample defaults with run overrides; this fixed script ignores
    them. The workflow's build_model(caller="workflow") receives the "workflow"
    entry. Real mode bypasses this factory and uses the configured provider.
    """
    return {"workflow": make_simulated_model()}
