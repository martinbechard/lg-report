"""Script model decisions, while DeepAgents executes the actual delegation.

Separate model instances give parent and specialist independent context meters
and response cursors. Sharing one would mix their histories and token counts.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# LangChain's AIMessage holds an assistant response; constructing it runs nothing.
# Its tool_calls, when present, are proposed names/arguments, not tool results.
from langchain_core.messages import AIMessage

from lg_report.platform.simulated_model import MeteredDemoModel

# The assignment must stand alone because the specialist starts without the
# parent's conversation. Only the final summary crosses back to the parent.
DELEGATED_TASK = "Look up ReAct in the workflow reference and summarize the loop and the role of tool observations."
# A compact child answer is the intended handoff, not its full tool transcript.
SPECIALIST_SUMMARY = "ReAct alternates model decisions with tool observations. A tool supplies evidence; the next model request includes that evidence before the answer is composed."
# The parent synthesizes the handoff into the user-facing response.
FINAL_ANSWER = "The specialist confirmed that ReAct repeats a decision, a tool action, and an observation. The parent delegates the lookup, receives a concise summary, and uses it to answer; the specialist's internal tool exchange stays in its own context."


def make_simulated_models() -> tuple[MeteredDemoModel, MeteredDemoModel]:
    """Make delegation reproducible while keeping parent and specialist usage separate.

    Return (parent_model, specialist_model), each with its own response ledger.

    Create a new pair for each run. Sharing one object would let the child consume
    the parent's next answer and incorrectly treat separate contexts as one cache.
    Construction is local; counts are measured when the real graph invokes each
    fixture. The fixtures request tools but do not execute those tools themselves.
    """
    parent_model = MeteredDemoModel(
        metadata={
            "report_effort": "light",
            "report_description": "Delegate reference research to the specialist, then answer using its summary.",
        },
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        # DeepAgents resolves this native tool name and uses
                        # subagent_type to select the registered specialist.
                        "name": "task",
                        "args": {
                            "subagent_type": "workflow-specialist",
                            "description": DELEGATED_TASK,
                        },
                        "id": "delegate-1",
                    }
                ],
            ),
            AIMessage(content=FINAL_ANSWER),
        ],
    )
    specialist_model = MeteredDemoModel(
        metadata={
            "report_effort": "light",
            "report_description": "Look up workflow evidence in an isolated context and return a concise summary to the parent.",
        },
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "workflow_reference",
                        "args": {"topic": "ReAct"},
                        "id": "specialist-lookup-1",
                    }
                ],
            ),
            AIMessage(content=SPECIALIST_SUMMARY),
        ],
    )
    return parent_model, specialist_model


# The client supplies this scenario; neither agent imports it.
USER_PROMPTS = [
    "Explain ReAct and how a specialist's tool observations help a parent agent answer."
]

# The client owns the question. The parent sees it first, while only the
# delegated task and specialist summary cross the explicit handoff boundary.
