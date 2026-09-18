"""A local reference lookup: no network, model request, or external fee.

The tool docstring describes its capability to the LLM. The return value becomes
a tool message in graph state, so it is included in the next LLM request.
"""

from langchain_core.tools import tool


@tool
def workflow_reference(topic: str) -> str:
    """Retrieve an explanation from a small local agent-workflow reference."""
    return (
        f"{topic}: an agent selects a tool, observes its result, and uses it to answer."
    )
