"""Route questions to the expert selected by the model, then relay its answer.

Only the workflow-supplied experts are exposed through DeepAgents' task tool.
Using that middleware directly avoids an implicit general-purpose fourth expert
and unrelated file tools. The agent does not import test questions or keywords.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent
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

# The dispatcher prompt defines a closed routing policy. Expert descriptions
# arrive through middleware, so registration metadata can be reviewed separately
# from the model-facing policy text.


def build_agent(parameters: dict) -> CompiledStateGraph:
    """Prepare a dispatcher so domain questions can reach the appropriate expert.

    parameters supplies the model and workflow-configured delegation middleware.
    The workflow registers expert graphs; the agent owns only its routing role.
    No domain data or user questions are supplied during construction. This
    function compiles the parent without executing either parent or children.

    Callback inheritance captures the child under task. The child returns a
    summary, not its private history. No manual child invoke or routing table
    based on question text is hidden in this function.
    """
    # The parent model selects a task name at runtime. Middleware executes the
    # corresponding child graph and makes its summary a ToolMessage in the
    # parent's history. The final parent invocation returns graph state, so a
    # caller reads the last answer from its messages rather than treating that
    # entire result as the child's tool output.
    # Register only the workflow's explicit children. Using the subagent
    # middleware directly avoids adding the general-purpose child and unrelated
    # tools that create_deep_agent would otherwise provide by default.
    return create_agent(
        **parameters,
        name="dispatcher_agent",
        system_prompt=SYSTEM_PROMPT,
    )
