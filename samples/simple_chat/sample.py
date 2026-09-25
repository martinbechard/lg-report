"""Author the simple chat example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. The shared catalog
constructs the offline model from CONVERSATION; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

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
