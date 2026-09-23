"""Define the parent that delegates the echo demonstration to isolated-subagent.

The task tool executes the specialist graph and returns its final summary.
No user scenarios, provider configuration, or recording belong in this module.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langgraph.graph.state import CompiledStateGraph

# The parent delegates the echo demonstration instead of duplicating the child's tools.
PARENT_PROMPT = "Delegate echoing the supplied text to isolated-subagent using task. Provide a self-contained assignment. After its summary returns, answer the user concisely. The specialist uses an echo tool, which supplies no independent factual evidence."

# The parent prompt is kept separate from the child specification so the
# delegation boundary remains inspectable and child evidence is not preloaded.


def build_agent(
    parameters: dict, specialist_specification: SubAgent
) -> CompiledStateGraph:
    """Prepare a parent that obtains the specialist's echo summary before answering the user.

    model handles the parent's decisions and final response. The specification
    contains the specialist's own model, instructions, and tools; DeepAgents
    compiles it and registers its name with task. This function makes no model
    request. The workflow's caller supplies messages when invoking the graph.
    """
    # A parent model request proposes a task call with an assignment. The task
    # middleware runs the child, then presents its final answer as the parent's
    # tool observation. The parent continues its own model loop to answer;
    # invoking this parent graph ultimately returns state containing messages.
    # Delegation is a normal graph tool operation, so nested callbacks retain
    # parent/child trace relationships without a second manual invocation.
    # DeepAgents compiles the specialist and preserves native delegation policy.
    return create_deep_agent(
        **parameters,
        name="delegating_parent",
        system_prompt=PARENT_PROMPT,
        subagents=[specialist_specification],
    )
