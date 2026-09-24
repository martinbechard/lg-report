"""Describe the example conversation in execution order, including the child exchange.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

The factory extracts assistant messages by speaker; the static client extracts
client messages. Tool entries document expected observations only: the graph
still executes the real tools and supplies their actual results.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "subagent_chat",
    "name": "Subagent chat",
    "description": "A parent agent delegates to a specialist.",
}


# "ai" belongs to the model created by the workflow and passed to the parent.
# A named speaker belongs to a model created by that agent itself. The order
# shows the nested conversation without requiring separate response arrays.
CONVERSATION = [
    {
        "role": "client",
        "content": "Have the specialist echo ReAct and explain how its tool result returns to the parent.",
    },
    # The specialist starts with this standalone assignment, not parent history.
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "task",
                "args": {
                    "subagent_type": "isolated-subagent",
                    "description": "Call echo_tool with the text ReAct and summarize what it returned.",
                },
                "id": "delegate-1",
            }
        ],
    },
    {
        "role": "isolated-subagent",
        "content": "",
        "tool_calls": [
            {
                "name": "echo_tool",
                "args": {"text": "ReAct"},
                "id": "specialist-lookup-1",
            }
        ],
    },
    {
        "role": "tool",
        "name": "echo_tool",
        "tool_call_id": "specialist-lookup-1",
        "content": "Your input was: ReAct",
    },
    {
        "role": "isolated-subagent",
        "content": "The echo tool returned Your input was: ReAct. It repeated the supplied text without adding factual evidence.",
    },
    {
        "role": "tool",
        "name": "task",
        "tool_call_id": "delegate-1",
        "content": "The echo tool returned Your input was: ReAct. It repeated the supplied text without adding factual evidence.",
    },
    {
        "role": "ai",
        "content": "The specialist echoed ReAct and received Your input was: ReAct. The parent receives its summary; the specialist's internal tool exchange stays in its own context.",
    },
]

USER_PROMPTS = client_prompts(CONVERSATION)
# Existing test helpers read the scenario; they do not author its text.
DELEGATED_TASK = CONVERSATION[1]["tool_calls"][0]["args"]["description"]
SPECIALIST_SUMMARY = CONVERSATION[4]["content"]
FINAL_ANSWER = CONVERSATION[-1]["content"]
