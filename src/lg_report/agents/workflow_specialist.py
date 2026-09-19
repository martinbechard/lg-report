"""Define the workflow specialist's isolated role and evidence tool.

DeepAgents compiles this specification under the parent's native task tool.
Only the delegated assignment enters its context; its final answer returns.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.middleware.subagents import SubAgent
from langchain_core.language_models import BaseChatModel

from lg_report.tools.workflow_reference import workflow_reference

# Only the assignment and this role instruction enter the isolated child context.
SPECIALIST_PROMPT = "You are a workflow reference specialist. Call workflow_reference for evidence about the assigned topic, then return a concise factual summary to the parent. Do not delegate further."

# This instruction describes the child's role. The parent supplies the
# assignment, while subagent middleware constructs the separate child context
# that owns this lookup and summary.


def build_agent(model: BaseChatModel) -> SubAgent:
    """Give the parent a specialist for obtaining concise workflow evidence.

    model is the LLM adapter retained by the specialist, not a compiled graph.
    Unlike standalone agents, this builder returns a DeepAgents specification:
    the parent's constructor supplies the standard child middleware. Neither
    specification creation nor compilation executes a model or evidence tool.
    """
    # This dictionary is configuration, not an invoked child or tool response.
    # Once registered, the parent's task tool invokes the child on an assignment
    # and returns its final answer as an observation for the parent to consume.
    # name is the task tool's routing identifier; description helps the parent
    # choose this role. system_prompt guides the child after selection. Keeping
    # the lookup tool here gives the child capabilities independent of its parent.
    return {
        "name": "workflow-specialist",
        "description": "Looks up agent workflow concepts and returns an evidence-based summary.",
        "system_prompt": SPECIALIST_PROMPT,
        "model": model,
        "tools": [workflow_reference],
    }
