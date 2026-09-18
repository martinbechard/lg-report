"""Script model decisions, while DeepAgents executes the actual delegation.

Separate model instances give parent and specialist independent context meters
and response cursors. Sharing one would mix their histories and token counts.
AI attribution: Generated with AI assistance.
"""

from langchain_core.messages import AIMessage

from lg_report.runner import MeteredDemoModel

DELEGATED_TASK = "Look up ReAct in the workflow reference and summarize the loop and the role of tool observations."
SPECIALIST_SUMMARY = "ReAct alternates model decisions with tool observations. A tool supplies evidence; the next model request includes that evidence before the answer is composed."
FINAL_ANSWER = "The specialist confirmed that ReAct repeats a decision, a tool action, and an observation. The parent delegates the lookup, receives a concise summary, and uses it to answer; the specialist's internal tool exchange stays in its own context."


def make_simulated_models() -> tuple[MeteredDemoModel, MeteredDemoModel]:
    """Return fresh parent and specialist models; counts are measured at invocation."""
    parent = MeteredDemoModel(
        metadata={
            "report_effort": "light",
            "report_description": "Delegate reference research to the specialist, then answer using its summary.",
        },
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
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
    specialist = MeteredDemoModel(
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
    return parent, specialist
