"""A local reference lookup: no network, model request, or external fee.

The tool docstring describes its capability to the LLM. The return value becomes
a tool message in graph state, so it is included in the next LLM request.

AI attribution: Generated with AI assistance.

Design: docs/chat-composition.md.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.tools import tool

# The decorator publishes the function's docstring as model-facing schema. Keep
# implementation rationale in comments so the teaching prompt remains stable.


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
    # Call this to supply the lesson's workflow explanation before the agent
    # composes its answer. The string below is the lookup's actual observation;
    # the consuming agent may use it in its own separately generated response.
    # Echoing the topic makes the tool message easy to correlate with the
    # assignment; it does not select a document or trigger a network/model call.
    return (
        f"{topic}: an agent selects a tool, observes its result, and uses it to answer."
    )
