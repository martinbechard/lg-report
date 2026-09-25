"""Author the simple chat example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.simulated_model import SimulatedModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "simple_chat",
    "name": "Simple chat",
    "description": "A direct conversation with a Deep Agent.",
}

CONVERSATION = [
    {"role": "client", "content": "Explain the main steps in an agent workflow."},
    {
        "role": "chat_agent",
        "content": "An agent observes its input, selects an action, uses the result, and responds. This report was "
        "captured from a real DeepAgents graph using an offline model.",
        "response_metadata": {"model_name": "scripted-chat"},
    },
    {"role": "client", "content": "How does a tool observation help the agent answer?"},
    {
        "role": "chat_agent",
        "content": "A tool observation supplies evidence the model did not have in the user's request. The model "
        "incorporates that evidence into its next answer.",
        "response_metadata": {"model_name": "scripted-chat", "finish_reason": "stop"},
    },
]


def build_scripted_models(options):
    """Create a fresh standard simulator with this lesson's report metadata.

    Response routing and token accounting use the shared SimulatedModel. This
    fixed conversation ignores options; the callback preserves only the report
    annotations. Live mode uses the configured provider instead.
    """
    return {
        "workflow": SimulatedModel(
            conversation=CONVERSATION,
            metadata={
                "report_effort": "light",
                "report_description": "Answer the current user question directly without "
                "requesting a tool.",
                "lc_versions": {"langchain-core": "1.6.3", "langchain": "1.4.1"},
            },
        )
    }
