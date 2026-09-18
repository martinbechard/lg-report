"""Define the parent that delegates evidence gathering to workflow-specialist.

The task tool executes the specialist graph and returns its final summary.
No user scenarios, provider configuration, or recording belong in this module.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.subagents import SubAgent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

# The parent delegates evidence gathering instead of duplicating the child's tools.
PARENT_PROMPT = "Delegate the workflow lookup to workflow-specialist using task. Provide a self-contained assignment. After its summary returns, answer the user concisely. Do not perform the lookup yourself."


def build_agent(
    model: BaseChatModel, specialist_specification: SubAgent
) -> CompiledStateGraph:
    """Build the parent around an LLM adapter and a child-role specification.

    model handles the parent's decisions and final response. The specification
    contains the specialist's own model, instructions, and tools; DeepAgents
    compiles it and registers its name with task. This function makes no model
    request. The workflow's caller supplies messages when invoking the graph.
    """
    # Delegation is a normal graph tool operation, so nested callbacks retain
    # parent/child trace relationships without a second manual invocation.
    # StateBackend confines built-in file tools to conversation state.
    return create_deep_agent(
        model=model,
        backend=StateBackend(),
        name="delegating_parent",
        system_prompt=PARENT_PROMPT,
        subagents=[specialist_specification],
    )
