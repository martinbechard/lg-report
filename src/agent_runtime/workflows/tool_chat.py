"""Select the reference_chat_agent role for the echo tool workflow.

Applications choose a model and a client; this module chooses the agent. Keeping
that decision here gives single-agent and multi-agent samples the same entry
point, so a client does not need to understand the graph's participants.
Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agent_runtime.agents.reference_chat_agent import build_agent
from agent_runtime.harness.model_factory import build_model


def build_workflow(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Let the application answer questions with the echo tool agent.

    Return a compiled graph ready for later conversation turns.

    model is an LLM adapter (live provider or scripted test model), not a list
    of messages. The graph retains it for later requests. The harness supplies
    user messages, conversation history, and tracing callbacks at invocation;
    building the workflow makes no provider request.
    """
    # Model selection belongs to the factory; the workflow supplies its role.
    if model is None:
        model = build_model(caller="workflow")
    # Select the tool-capable role without importing its tool or prompt.
    # The model inside that agent chooses when to request an echo; a workflow
    # that pre-called the tool would teach application-driven execution instead.
    return build_agent({"model": model, "backend": StateBackend()})
