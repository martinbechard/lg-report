"""Select the chat_agent role for the direct conversation workflow.

Applications choose a model and a client; this module chooses the agent. Keeping
that decision here gives single-agent and multi-agent samples the same entry
point, so a client does not need to understand the graph's participants.
Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.agents.chat_agent import build_agent


def build_workflow(model: BaseChatModel) -> CompiledStateGraph:
    """Let the application offer direct conversation through the shared workflow API.

    Return a compiled graph ready for later conversation turns.

    model is an LLM adapter (live provider or scripted test model), not a list
    of messages. The graph retains it for later requests. The platform supplies
    user messages, conversation history, and tracing callbacks at invocation;
    building the workflow makes no provider request.
    """
    return build_agent(model)
