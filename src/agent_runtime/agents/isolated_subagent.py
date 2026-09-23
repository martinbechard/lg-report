"""Define the isolated subagent's role and echo tool.

DeepAgents compiles this specification under the parent's native task tool.
Only the delegated assignment enters its context; its final answer returns.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.middleware.subagents import SubAgent

from agent_runtime.harness.model_factory import build_model
from agent_runtime.tools.echo_tool import echo_tool

# Only the assignment and this role instruction enter the isolated child context.
SPECIALIST_PROMPT = "You are a workflow teaching specialist. Call echo_tool with the assigned text, then summarize the returned echo for the parent. The echo repeats supplied input and provides no independent factual evidence. Do not delegate further."

# This instruction describes the child's role. The parent supplies the
# assignment, while subagent middleware constructs the separate child context
# that owns this echo and summary.


def build_agent(parameters: dict | None = None) -> SubAgent:
    """Give the parent a specialist for demonstrating a tool call in an isolated context.

    The specialist obtains its own model by default. Optional parameters allow
    other workflows to supply a model and additional middleware.
    Unlike standalone agents, this builder returns a DeepAgents specification:
    DeepAgents compiles the child using its native middleware and supplied policy. Neither
    specification creation nor compilation executes a model or echo tool.
    """
    parameters = dict(parameters or {})
    # Preserve explicit models used by other lessons without constructing an
    # unused provider client (which would also require credentials).
    if "model" not in parameters:
        parameters["model"] = build_model(caller="isolated-subagent")
    # This dictionary is configuration, not an invoked child or tool response.
    # Once registered, the parent's task tool invokes the child on an assignment
    # and returns its final answer as an observation for the parent to consume.
    # name is the task tool's routing identifier; description helps the parent
    # choose this role. system_prompt guides the child after selection. Keeping
    # the echo tool here gives the child capabilities independent of its parent.
    return {
        "name": "isolated-subagent",
        "description": "Echoes assigned text with a local tool and summarizes its result.",
        "system_prompt": SPECIALIST_PROMPT,
        **parameters,
        "tools": [echo_tool],
    }
