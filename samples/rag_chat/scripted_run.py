"""Supply an offline model fixture while still executing real Chroma retrieval.

The fixture preselects a citation and excerpt from the built index so it never
pretends to generate an unscripted answer. Live mode replaces this model entirely.
This tests retrieval plumbing and accounting, not live answer quality.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# LangChain's AIMessage holds an assistant response; constructing it runs nothing.
# Its tool_calls, when present, are proposed names/arguments, not tool results.
from langchain_core.messages import AIMessage

from agent_runtime.agents.wikipedia_rag_agent import WIKIPEDIA_INDEX_DIRECTORY
from agent_runtime.harness.rag_index import open_index
from agent_runtime.harness.simulated_model import MeteredDemoModel

USER_PROMPTS = ["How has the Australian raven adapted to urban environments?"]


def make_simulated_model():
    """Make the retrieval lesson repeatable with a citation from the actual index.

    Return a scripted model whose final answer contains a preselected excerpt.

    This preparatory local search selects the repeatable fixture's expected output;
    the agent still executes its own recorded search during the actual workflow.
    """
    # Only the offline fixture performs this preparatory lookup to author its
    # expected response. Opening may restore the packaged index on first use;
    # an absent/incomplete index raises. The app/workflow never pass storage
    # into the agent.
    collection = open_index(WIKIPEDIA_INDEX_DIRECTORY)
    result = collection.query(query_texts=USER_PROMPTS, n_results=1)
    # This is Chroma's direct query result, not an agent invocation result.
    # The outer list identifies our first query; the inner list its first match.
    # An empty index cannot provide the required excerpt and fails here.
    passage_id = result["ids"][0][0]
    excerpt = result["documents"][0][0][:600]
    return MeteredDemoModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search_wikipedia",
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


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}
