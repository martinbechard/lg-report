"""Connect the delegating parent to a workflow-reference specialist.

The workflow chooses participants; agent modules own instructions and tools.
The specialist is registered as a specification, which DeepAgents compiles when
building the parent. Actual delegation happens later through the task tool.
Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.agents.delegating_parent import build_agent as build_parent
from lg_report.agents.workflow_specialist import build_agent as build_specialist


def build_workflow(
    parent_model: BaseChatModel, specialist_model: BaseChatModel
) -> CompiledStateGraph:
    """Return an uninvoked parent graph with one registered specialist role.

    Both arguments are LLM adapters, not agents or message collections. The
    application provides separate instances so scripted response positions and
    simulated token histories do not leak between roles. The instances may use
    the same provider model; specialization comes from instructions and tools.
    """
    # This dictionary describes a child; it is not the child's response or an
    # already-running task. The parent builder lets DeepAgents compile it and
    # associate its name with the task tool's delegation choices.
    specialist_specification = build_specialist(specialist_model)
    return build_parent(parent_model, specialist_specification)
