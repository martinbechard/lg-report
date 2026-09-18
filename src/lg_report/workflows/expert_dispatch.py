"""Connect a dispatcher to movie, sports, and history expert graphs.

The application supplies one shared LLM, not domain data or user questions.
This workflow gives that LLM to every agent and registers the experts with the
dispatcher. Role instructions remain in the individual agent files.
See docs/chat-composition.md for the composition and delegation diagrams.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.agents import (
    dispatcher_agent,
    history_expert,
    movie_expert,
    sports_expert,
)


def build_workflow(model: BaseChatModel) -> CompiledStateGraph:
    """Build all four agents with one shared LLM.

    model is the single configured provider adapter or offline simulator. Each
    agent combines it with its own instructions, tools, and graph-owned message
    history. Sharing the LLM does not merge those conversations.
    Construction makes no model requests; Conversation invokes the returned graph.
    """
    # Expertise comes from the role's instructions and retrieval tool. All four
    # agents use this same LLM object; there are no per-expert model settings.
    movie_graph = movie_expert.build_agent(model)
    sports_graph = sports_expert.build_agent(model)
    history_graph = history_expert.build_agent(model)

    # DeepAgents needs three things to make an expert available through `task`:
    # name is the identifier the dispatcher puts in subagent_type;
    # description tells the dispatcher when to choose that expert;
    # runnable is the already-built graph to execute when that name is selected.
    # These registrations tell the task tool which graph to invoke.
    experts: list[CompiledSubAgent] = [
        {
            "name": movie_expert.NAME,
            "description": movie_expert.DESCRIPTION,
            "runnable": movie_graph,
        },
        {
            "name": sports_expert.NAME,
            "description": sports_expert.DESCRIPTION,
            "runnable": sports_graph,
        },
        {
            "name": history_expert.NAME,
            "description": history_expert.DESCRIPTION,
            "runnable": history_graph,
        },
    ]

    # The dispatcher uses the same LLM. Its task middleware invokes only the expert
    # it selects and returns that expert's answer to the dispatcher conversation.
    return dispatcher_agent.build_agent(model, experts)
