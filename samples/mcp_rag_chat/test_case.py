"""Define a scripted MCP search scenario without opening the database directly.

The real MCP server supplies evidence during execution. The fixed final response
only acknowledges retrieval; it does not pretend to synthesize a factual answer.
Live mode replaces this fixture with a configured provider model.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# AI attribution: Modified with AI assistance.
# This fixture cannot prove MCP answer quality because its final text is canned;
# it proves that a real search result crosses the tool boundary.
# LangChain's AIMessage holds an assistant response; constructing it runs nothing.
# Its tool_calls, when present, are proposed names/arguments, not tool results.
from langchain_core.messages import AIMessage

from lg_report.platform.simulated_model import MeteredDemoModel

USER_PROMPTS = ["How has the Australian raven adapted to urban environments?"]

# The question is kept outside the workflow so the sample demonstrates how a
# client supplies user input. The workflow remains reusable for other prompts.


def make_simulated_model():
    """Exercise the MCP retrieval path with predictable model decisions.

    Return a fixture proposing one search and then emitting a canned
    acknowledgement; constructing it does not contact the MCP server.

    Returns:
        A metered offline model whose tool request exercises the real MCP
        boundary and whose final text makes the fixture limitation explicit.

    Side effects:
        None during construction; MCP I/O occurs when the graph executes the
        scripted search request.
    """
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
