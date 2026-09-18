"""Define a scripted MCP search scenario without opening the database directly.

The real MCP server supplies evidence during execution. The fixed final response
only acknowledges retrieval; it does not pretend to synthesize a factual answer.
Live mode replaces this fixture with a configured provider model.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from lg_report.platform.simulated_model import MeteredDemoModel

USER_PROMPTS = ["How has the Australian raven adapted to urban environments?"]


def make_simulated_model():
    """Request one real MCP search, then acknowledge the observed tool result."""
    return MeteredDemoModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_wikipedia",
                        "args": {"query": USER_PROMPTS[0]},
                        "id": "wiki-mcp-search-1",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "The MCP search completed. Inspect the search_wikipedia result "
                    "for retrieved passages and source IDs. This scripted response "
                    "demonstrates retrieval and accounting, not answer synthesis."
                )
            ),
        ],
        metadata={
            "report_description": "Search Wikipedia through MCP and consume the returned evidence.",
            "report_effort": "light",
        },
    )
