"""Declare a teaching variant that shares another sample's implementation.

The catalog uses SAMPLE for selection and loads conversation/model factories
from the declared implementation. Importing this file does not build a model.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    # The adaptive test user evaluates this goal before requesting another turn.
    "goal": 'The user understands the main steps of an agent workflow AND how a tool observation supplies evidence for the next action or final answer. Both topics must be explained clearly, with an example of using a tool observation.',
    "id": "simple_chat_langfuse",
    "name": "Simple chat — Langfuse",
    "description": "Record conversation traces in configured Langfuse.",
    "tracing": "langfuse",
    "implementation": "simple_chat",
}
