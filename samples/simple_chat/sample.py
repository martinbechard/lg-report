"""Author the simple chat example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts
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

# User inputs and model replies are projections of the same ordered script.
USER_PROMPTS = client_prompts(CONVERSATION)


def make_simulated_model():
    """Extract AI replies into a fresh model with independent cursor and usage."""
    return SimulatedModel(
        conversation=CONVERSATION,
        metadata={
            "report_effort": "light",
            "report_description": "Answer the current user question directly without "
            "requesting a tool.",
            "lc_versions": {"langchain-core": "1.6.3", "langchain": "1.4.1"},
        },
    )


def build_scripted_models(options):
    """Let the catalog retain this sample's report metadata in simulated mode.

    The catalog explicitly calls this callback when building a simulated run.
    Importing the module does not call it. Each call creates a fresh response
    cursor and usage ledger so independent conversations start at the beginning.
    options merges sample defaults with run overrides; this fixed script ignores
    them. The workflow's build_model(caller="workflow") receives the "workflow"
    entry. Real mode bypasses this factory and uses the configured provider.
    """
    return {"workflow": make_simulated_model()}
