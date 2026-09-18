"""Own the simple-chat user prompts and scripted model answers as one test case.

Prompts and response content are scripted; token counts are not. MeteredDemoModel
measures the actual growing message history so the test case does not maintain
a second set of token totals.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from lg_report.platform.simulated_model import MeteredDemoModel

# Only the static client consumes these prompts; the agent receives them at
# invocation time, exactly as it receives console input.
USER_PROMPTS = [
    "Explain the main steps in an agent workflow.",
    "How does a tool observation help the agent answer?",
]


def make_simulated_model() -> MeteredDemoModel:
    """Return a new two-answer fixture with an empty context/cache ledger.

    Create one instance per conversation, then reuse it across its turns.
    Responses are predetermined, but input usage is measured from the messages
    the graph actually supplies; these counts are illustrative, not provider
    tokenizer results. Constructing this fixture makes no network requests.
    """
    # These annotations explain the operation in reports. They are not provider
    # reasoning controls and do not constitute hidden model reasoning.
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
    # The second answer belongs to the next user turn, not to a tool loop.
    # No tool_calls in the first answer means the graph can finish that turn.
    model.responses.append(
        AIMessage(
            content="A tool observation supplies evidence the model did not have in the user's request. The model incorporates that evidence into its next answer.",
            response_metadata={"model_name": "scripted-chat", "finish_reason": "stop"},
        )
    )
    return model
