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
    """Build the Wikipedia agent; its completed local index must already exist."""
    return build_agent(model)
