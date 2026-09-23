"""Connect the delegating parent to a workflow teaching specialist.

The workflow chooses participants; agent modules own instructions and tools.
The specialist is registered as a specification, which DeepAgents compiles when
building the parent. Actual delegation happens later through the task tool.
Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.backends import StateBackend
from langgraph.graph.state import CompiledStateGraph

from agent_runtime.agents.delegating_parent import build_agent as build_parent
from agent_runtime.agents.isolated_subagent import build_agent as build_specialist
from agent_runtime.harness.model_factory import build_model


def build_workflow() -> CompiledStateGraph:
    """Compose a parent with a specialist that owns its own model selection.

    The workflow obtains the parent's model using the configured model name.
    The factory resolves provider settings; specialization comes from each
    agent's instructions and tools. Construction does not invoke either agent.
    """
    # Agent delegation flow (DeepAgents owns the internal model/tool graph):
    #
    # input messages -> delegating_parent -> final answer
    #                         |  ^
    #        task(assignment) |  | specialist's final answer as tool result
    #                         v  |
    #                  isolated-subagent
    #                         |  ^
    #               echo_tool |  | echo result
    #                         v  |
    #                      echo_tool
    #
    # These arrows describe calls and returns, not a custom StateGraph's edges.
    # The parent chooses task calls at runtime and continues after each return;
    # its prompt requests delegation, but construction does not force a call.
    # The specialist receives a self-contained assignment in isolated context,
    # not the parent's conversation. echo_tool is a tool, not another agent.
    parent_model = build_model(caller="workflow")
    # The child owns its model and tools. DeepAgents compiles this specification
    # and later invokes it when the parent requests its registered task name.
    specialist_specification = build_specialist()
    return build_parent(
        {"model": parent_model, "backend": StateBackend()},
        specialist_specification,
    )
