"""Scripted responses keep the chat exercise reproducible without an API key.

Only response content is scripted. MeteredDemoModel measures the actual growing
message history so the sample does not maintain a second set of token totals.
"""

from langchain_core.messages import AIMessage

from lg_report.runner import MeteredDemoModel


def make_simulated_model() -> MeteredDemoModel:
    model = MeteredDemoModel(
        metadata={
            "report_effort": "light",
            "report_description": "Answer the current user question directly without requesting a tool.",
        },
        responses=[
            AIMessage(
                content="An agent observes its input, selects an action, uses the result, and responds. "
                "This report was captured from a real DeepAgents graph using an offline model.",
                response_metadata={"model_name": "scripted-chat"},
            )
        ],
    )
    model.responses.append(
        AIMessage(
            content="A tool observation supplies evidence the model did not have in the user's request. The model incorporates that evidence into its next answer.",
            response_metadata={"model_name": "scripted-chat", "finish_reason": "stop"},
        )
    )
    return model
