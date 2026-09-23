"""Define chat_agent independently of clients, test cases, and report recording.

The agent owns its instructions and graph. User messages arrive only at invoke;
it cannot import a predetermined conversation or decide how the client gets input.
AI attribution: Generated with AI assistance.

Design: docs/chat-composition.md.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

# This instruction keeps the first lesson focused on conversation history.
# DeepAgents retains its native tools; their definitions count toward input usage.
SYSTEM_PROMPT = (
    "Answer the user's chat question directly and concisely. "
    "Do not use tools for this simple chat exercise."
)

# This public prompt is a reviewable role contract. Keeping it as data lets
# samples inspect the lesson's instructions without constructing a graph.


def build_agent(parameters: dict) -> CompiledStateGraph:
    """Prepare a conversational assistant so clients can answer successive user turns.

    Keeping the model injectable lets the same application teach both a
    repeatable offline conversation and real provider usage. Compilation does
    not call the model; callers provide messages and callbacks when invoking it.
    """
    # Invoking the returned graph later accepts a state mapping with messages
    # and returns updated state. Its last AIMessage carries the chat answer;
    # the whole graph result is not a tool result or a single model message.
    # Use the native DeepAgents graph and retain its built-in behavior.
    # Reports include the tool definitions even when this lesson does not call them.
    return create_deep_agent(
        **parameters,
        name="chat_agent",
        system_prompt=SYSTEM_PROMPT,
    )
