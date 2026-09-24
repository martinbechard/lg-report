"""Author the tool chat example in chronological conversation order.

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
    "id": "tool_chat",
    "name": "Tool chat",
    "description": "Watch an agent echo your input through a tool.",
}

CONVERSATION = [
    {"role": "client", "content": "Echo this text: ReAct"},
    {
        "role": "reference_chat_agent",
        "content": "",
        "tool_calls": [
            {
                "name": "echo_tool",
                "args": {"text": "ReAct"},
                "id": "reference-1",
            }
        ],
        "response_metadata": {
            "model_name": "scripted-chat",
            "finish_reason": "tool_calls",
        },
    },
    {
        "role": "tool",
        "name": "echo_tool",
        "tool_call_id": "reference-1",
        "content": "Your input was: ReAct",
    },
    {
        "role": "reference_chat_agent",
        "content": 'The tool returned "Your input was: ReAct". This shows the model requesting a tool and then '
        "reading its result.",
        "response_metadata": {"model_name": "scripted-chat", "finish_reason": "stop"},
    },
    {"role": "client", "content": "Echo this text: Tool observations"},
    {
        "role": "reference_chat_agent",
        "content": "I will echo the text through the local tool.",
        "tool_calls": [
            {
                "name": "echo_tool",
                "args": {"text": "Tool observations"},
                "id": "reference-2",
            }
        ],
        "response_metadata": {
            "model_name": "scripted-chat",
            "finish_reason": "tool_calls",
        },
    },
    {
        "role": "tool",
        "name": "echo_tool",
        "tool_call_id": "reference-2",
        "content": "Your input was: Tool observations",
    },
    {
        "role": "reference_chat_agent",
        "content": 'The tool returned "Your input was: Tool observations". The next model request includes that '
        "result. This echo repeats supplied text and adds no independent evidence.",
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
            "report_effort": "fast",
            "report_description": "Request an echo, or compose the answer using the tool "
            "observation.",
            "lc_versions": {"langchain-core": "1.6.3", "langchain": "1.4.1"},
        },
    )


def build_scripted_models(options):
    """Let the catalog retain this sample's report metadata in simulated mode.

    The catalog explicitly calls this callback when building a simulated run.
    options merges sample defaults with run overrides; this fixed script ignores
    them. The workflow's build_model(caller="workflow") receives the "workflow"
    entry. Real mode bypasses this factory and uses the configured provider.
    """
    return {"workflow": make_simulated_model()}
