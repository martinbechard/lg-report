"""Echo caller-provided text to demonstrate the model/tool/message boundary.

This deterministic local tool makes no network or model request. Its return
value becomes a tool message that the next model request includes as input.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.tools import tool


# The docstring and typed argument form the model-facing tool schema. The tool
# supplies no independent evidence: it returns exactly the caller's text with
# a label, making the request/result relationship visible in teaching traces.
@tool(parse_docstring=True)
def echo_tool(text: str) -> str:
    """Return the provided text prefixed with "Your input was: ".

    Args:
        text: The string to echo unchanged after the prefix.
    """
    return f"Your input was: {text}"
