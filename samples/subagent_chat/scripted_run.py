"""Describe the example conversation in execution order, including the child exchange.

The factory extracts assistant messages by speaker; the static client extracts
client messages. Tool entries document expected observations only: the graph
still executes the real tools and supplies their actual results.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts

# The assignment must stand alone because the specialist starts without the
# parent's conversation. Only the final summary crosses back to the parent.
DELEGATED_TASK = "Call echo_tool with the text ReAct and summarize what it returned."
# A compact child answer is the intended handoff, not its full tool transcript.
SPECIALIST_SUMMARY = "The echo tool returned Your input was: ReAct. It repeated the supplied text without adding factual evidence."
# The parent synthesizes the handoff into the user-facing response.
FINAL_ANSWER = "The specialist echoed ReAct and received Your input was: ReAct. The parent receives its summary; the specialist's internal tool exchange stays in its own context."


# "ai" belongs to the model created by the workflow and passed to the parent.
# A named speaker belongs to a model created by that agent itself. The order
# shows the nested conversation without requiring separate response arrays.
CONVERSATION = [
    {
        "role": "client",
        "content": "Have the specialist echo ReAct and explain how its tool result returns to the parent.",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "task",
                "args": {
                    "subagent_type": "workflow-specialist",
                    "description": DELEGATED_TASK,
                },
                "id": "delegate-1",
            }
        ],
    },
    {
        "role": "workflow-specialist",
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
    {"role": "workflow-specialist", "content": SPECIALIST_SUMMARY},
    {
        "role": "tool",
        "name": "task",
        "tool_call_id": "delegate-1",
        "content": SPECIALIST_SUMMARY,
    },
    {"role": "ai", "content": FINAL_ANSWER},
]

USER_PROMPTS = client_prompts(CONVERSATION)
