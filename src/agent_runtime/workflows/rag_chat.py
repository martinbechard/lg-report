"""Select the Wikipedia RAG agent for this single-agent workflow.

The agent owns its fixed Chroma collection and search tool. This composition
layer selects the role and requests its LLM from build_model. The harness
selects the user client.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agent_runtime.agents.wikipedia_rag_agent import build_agent
from agent_runtime.harness.model_factory import build_model


def build_workflow(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Let the application answer questions using the local Wikipedia corpus.

    ``model`` is the configured live or scripted chat model. Return a compiled
    agent graph whose search tool retrieves passages for later model calls.
    Building may restore the packaged index into the default directory when
    that directory is absent or empty, writing local files before opening and
    validating the collection. Unavailable/unrestorable or incomplete storage
    fails here; construction does not invoke the model or ingest the corpus.
    """
    # Model selection belongs to the factory; the workflow supplies its role.
    if model is None:
        model = build_model(caller="workflow")
    # The role encapsulates which collection to open and how to expose retrieval.
    # Construction may prepare storage, but opening an index is not a search:
    # passages enter model context only after an agent-selected tool call.
    return build_agent({"model": model})
