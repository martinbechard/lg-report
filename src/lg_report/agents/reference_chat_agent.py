"""Define an agent that grounds answers in the local workflow reference.

Own instructions and tool registration; clients supply user requests at runtime.
The injected model can be a provider or a deterministic test fixture.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.tools.workflow_reference import workflow_reference

# Role instructions apply to arbitrary user questions, not a fixed test case.
SYSTEM_PROMPT = "Use workflow_reference to obtain evidence for each question about agent workflows, then answer concisely using the observation."


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Return a graph that can obtain local evidence before answering.

    model is either the scripted fixture or a configured provider adapter.
    Graph construction does not invoke either. The returned graph owns tool
    routing; callers own conversation history and tracing callbacks.
    """
    # Register the actual function, not a fabricated tool-result string. This is
    # what makes the resulting trace a runnable tool-use example rather than a
    # preassembled report. LangGraph owns the model/tool/model routing.
    return create_deep_agent(
        model=model,
        tools=[workflow_reference],
        backend=StateBackend(),
        name="reference_chat_agent",
        system_prompt=SYSTEM_PROMPT,
    )
