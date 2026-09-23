"""Define an agent that demonstrates calling a local echo tool.

Own instructions and tool registration; clients supply user requests at runtime.
The injected model can be a provider or a deterministic test fixture.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

from agent_runtime.tools.echo_tool import echo_tool

# Role instructions apply to arbitrary user questions, not a fixed test case.
SYSTEM_PROMPT = "Call echo_tool with the user's text, then answer concisely using the returned echo. The tool repeats input; it provides no independent factual evidence."

# This prompt makes echoing a deliberate tool step so the model consumes a
# real observation before answering. The echo is not a reference lookup.


def build_agent(parameters: dict) -> CompiledStateGraph:
    """Prepare an agent that demonstrates an echo and its resulting tool message.

    parameters supplies the scripted or live model and workflow execution policy.
    Graph construction does not invoke either. The returned graph owns tool
    routing; callers own conversation history and tracing callbacks. Model and
    tool failures remain visible to the invoking workflow; this constructor
    provides no fallback evidence.
    """
    # At execution, an AIMessage.tool_calls entry proposes the echo arguments.
    # The graph dispatches the registered tool and records its returned string
    # as a LangChain ToolMessage, which the next model call can read. The final
    # invocation result is graph state containing that exchange and the answer.
    # Register the actual function, not a fabricated tool-result string. This is
    # what makes the resulting trace a runnable tool-use example rather than a
    # preassembled report. LangGraph owns the model/tool/model routing.
    return create_deep_agent(
        **parameters,
        tools=[echo_tool],
        name="reference_chat_agent",
        system_prompt=SYSTEM_PROMPT,
    )
