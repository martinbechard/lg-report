"""A local reference lookup: no network, model request, or external fee.

The tool docstring describes its capability to the LLM. The return value becomes
a tool message in graph state, so it is included in the next LLM request.

AI attribution: Generated with AI assistance.

Design: docs/chat-composition.md.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.tools import tool


# @tool derives the model-facing tool description from the function docstring;
# changing that docstring changes prompt content and its measured input size.
# topic labels the fixed explanation; this fixture does not perform semantic
# search or choose among documents. It teaches the model/tool/message boundary.
# The returned string becomes paid input only when the next model consumes it;
# executing this deterministic local function itself makes no model call.
@tool(parse_docstring=True)
def workflow_reference(topic: str) -> str:
    """Return a fixed explanation from a small local agent-workflow reference.

    Args:
        topic: The workflow concept used to label the explanation, such as tool
            calling or delegation. This fixture returns the same explanation
            for every topic.
    """
    return (
        f"{topic}: an agent selects a tool, observes its result, and uses it to answer."
    )
