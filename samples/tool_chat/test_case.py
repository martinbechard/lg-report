"""Script two model/tool/model cycles to teach the complete tool-call lifecycle.

Each tool request has a distinct ID: LangGraph matches its result to that ID.
The graph executes the real local tool; this fixture scripts only the model.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from lg_report.platform.simulated_model import MeteredDemoModel


def make_simulated_model() -> MeteredDemoModel:
    """Return a fresh four-response fixture for two model/tool/model cycles.

    Reuse this model across both user turns so its response cursor and context
    ledger stay aligned. Only decisions and answers are scripted: the graph
    executes workflow_reference and feeds its actual result to the next call.
    No provider requests occur while constructing or using this fixture.
    """
    # A tool-call response can have empty visible text. The structured call
    # still counts as model output and must be retained in conversation history.
    model = MeteredDemoModel(
        metadata={
            "report_effort": "fast",
            "report_description": "Decide whether a reference is needed, or compose the answer using the tool observation.",
        },
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "workflow_reference",
                        "args": {"topic": "ReAct"},
                        "id": "reference-1",
                    }
                ],
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "tool_calls",
                },
            ),
            AIMessage(
                content="The model requested a local workflow reference. The tool returned an explanation, and the model used that observation to produce this answer.",
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "stop",
                },
            ),
        ],
    )
    # This turn also demonstrates visible assistant text alongside a tool call.
    # Both belong to one model response; the tool result is a separate message.
    model.responses.extend(
        [
            AIMessage(
                content="I will look up how observations contribute to an answer.",
                tool_calls=[
                    {
                        "name": "workflow_reference",
                        "args": {"topic": "Tool observations"},
                        "id": "reference-2",
                    }
                ],
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "tool_calls",
                },
            ),
            AIMessage(
                content="The tool observation provides external evidence. After the lookup, the model reads that result, connects it to your question, and produces a grounded answer. Here the evidence came from the local workflow reference.",
                response_metadata={
                    "model_name": "scripted-chat",
                    "finish_reason": "stop",
                },
            ),
        ]
    )
    return model


# User input belongs to the test scenario, never the agent definition.
USER_PROMPTS = [
    "Explain the main steps in an agent workflow.",
    "How does a tool observation help the agent answer?",
]
