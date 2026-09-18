"""Route questions to the expert selected by the model, then relay its answer.

Only the workflow-supplied experts are exposed through DeepAgents' task tool.
Using that middleware directly avoids an implicit general-purpose fourth expert
and unrelated file tools. The agent does not import test questions or keywords.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.backends import StateBackend
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

# Descriptions on registered experts explain their domains; these instructions
# govern when to delegate versus ask for clarification or decline a mismatch.
SYSTEM_PROMPT = (
    "You are a dispatcher for movies, sports, and history. For each question, "
    "select the most relevant expert from the task tool descriptions and delegate "
    "a self-contained question to exactly that expert. Include relevant context "
    "from earlier turns when needed. Use the expert's answer to respond; do not "
    "answer domain questions yourself. For ambiguous questions ask the user which "
    "domain they mean. For requests outside these domains explain the supported "
    "subjects without delegating. Never choose an expert merely to force a match."
)


def build_agent(
    model: BaseChatModel, experts: list[CompiledSubAgent]
) -> CompiledStateGraph:
    """Register the workflow's compiled experts with native isolated delegation.

    model is the dispatcher LLM adapter. experts contains registration records
    with a routing name, capability description, and already-compiled runnable
    graph for each child. It contains no domain data or user questions. This
    function compiles the parent without executing either parent or children.

    Callback inheritance captures the child under task. The child returns a
    summary, not its private history. No manual child invoke or routing table
    based on question text is hidden in this function.
    """
    # Register only the workflow's explicit children. Using the subagent
    # middleware directly avoids adding the general-purpose child and unrelated
    # tools that create_deep_agent would otherwise provide by default.
    return create_agent(
        model=model,
        name="dispatcher_agent",
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=experts,
                task_description="Delegate a self-contained question to the matching expert. Available experts:\n{available_agents}",
            )
        ],
    )
