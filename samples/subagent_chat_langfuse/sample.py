"""Declare a teaching variant that shares another sample's implementation.

The catalog uses SAMPLE for selection and loads conversation/model factories
from the declared implementation. Importing this file does not build a model.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    # The adaptive test user evaluates this goal before requesting another turn.
    "goal": 'The specialist has echoed ReAct and the parent has explained how the specialist tool result returns to the parent, supported by the reported result.',
    "id": "subagent_chat_langfuse",
    "name": "Subagent chat — Langfuse",
    "description": "Record parent and specialist traces in configured Langfuse.",
    "tracing": "langfuse",
    "implementation": "subagent_chat",
}
