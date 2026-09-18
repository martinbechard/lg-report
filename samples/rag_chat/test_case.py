"""Supply an offline model fixture while still executing real Chroma retrieval.

The fixture preselects a citation and excerpt from the built index so it never
pretends to generate an unscripted answer. Live mode replaces this model entirely.
This tests retrieval plumbing and accounting, not live answer quality.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from lg_report.agents.wikipedia_rag_agent import WIKIPEDIA_INDEX_DIRECTORY
from lg_report.platform.rag_index import open_index
from lg_report.platform.simulated_model import MeteredDemoModel

USER_PROMPTS = ["How has the Australian raven adapted to urban environments?"]


def make_simulated_model():
    """Author a search request and source excerpt using real indexed evidence.

    This preparatory local search selects the repeatable fixture's expected output;
    the agent still executes its own recorded search during the actual workflow.
    """
    # Only the offline fixture performs this preparatory lookup to author its
    # expected response. The app/workflow never pass storage into the agent.
    collection = open_index(WIKIPEDIA_INDEX_DIRECTORY)
    result = collection.query(query_texts=USER_PROMPTS, n_results=1)
    passage_id = result["ids"][0][0]
    excerpt = result["documents"][0][0][:600]
    return MeteredDemoModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_wikipedia",
                        "args": {"query": USER_PROMPTS[0]},
                        "id": "wiki-search-1",
                    }
                ],
            ),
            AIMessage(
                content=f"Retrieved reference excerpt [{passage_id}]:\n{excerpt}"
            ),
        ],
        metadata={
            "report_description": "Retrieve bounded evidence and answer with passage citations.",
            "report_effort": "light",
        },
    )
