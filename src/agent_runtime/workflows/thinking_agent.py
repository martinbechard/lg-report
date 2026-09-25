"""Select the investigation_agent role for the service investigation workflow.

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

from agent_runtime.agents.investigation_agent import build_agent
from agent_runtime.harness.model_factory import build_model


def build_workflow(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Let the application investigate service problems with its investigation agent.

    Return a compiled graph ready for later conversation turns.

    model is an LLM adapter (live provider or scripted test model), not a list
    of messages. The graph retains it for later requests. The harness supplies
    user messages, conversation history, and tracing callbacks at invocation;
    building the workflow makes no provider request.
    """
    # Ask for a capability role so deployment choices stay in configuration.
    # The live factory resolves LG_MODEL_ADVANCED once; demo mode stays offline.
    if model is None:
        model = build_model(symbolic_model_name="advanced", caller="workflow")
    # Choose the investigator as one component. Its tool loop decides which
    # evidence to inspect and when to test a plan; this layer must not hard-code
    # the offline fixture's sequence or turn that sequence into live routing.
    return build_agent({"model": model, "backend": StateBackend()})
