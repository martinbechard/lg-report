"""Select the Wikipedia RAG agent for this single-agent workflow.

The agent owns its fixed Chroma collection and search tool. This composition
layer selects the role; the application supplies the LLM and user client.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.agents.wikipedia_rag_agent import build_agent


def build_workflow(model: BaseChatModel) -> CompiledStateGraph:
    """Let the application answer questions using the local Wikipedia corpus.

    ``model`` is the configured live or scripted chat model. Return a compiled
    agent graph whose search tool retrieves passages for later model calls.
    Building may restore the packaged index into the default directory when
    that directory is absent or empty, writing local files before opening and
    validating the collection. Unavailable/unrestorable or incomplete storage
    fails here; construction does not invoke the model or ingest the corpus.
    """
    return build_agent(model)
