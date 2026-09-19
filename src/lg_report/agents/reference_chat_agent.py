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

# This prompt makes the local lookup a deliberate evidence step. The fixture
# must not be described as internet access or broader reference coverage.


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Prepare a workflow tutor that can ground its answers in local reference text.

    model is either the scripted fixture or a configured provider adapter.
    Graph construction does not invoke either. The returned graph owns tool
    routing; callers own conversation history and tracing callbacks. Model and
    tool failures remain visible to the invoking workflow; this constructor
    provides no fallback evidence.
    """
    # At execution, an AIMessage.tool_calls entry proposes the lookup arguments.
    # The graph dispatches the registered tool and records its returned string
    # as a LangChain ToolMessage, which the next model call can read. The final
    # invocation result is graph state containing that exchange and the answer.
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
