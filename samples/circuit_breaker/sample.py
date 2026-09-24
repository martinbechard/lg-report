"""Describe the circuit-breaker conversation in the order its steps occur.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

User messages, scripted AI explanations and tool calls, and expected tool results
share one chronological list. Several AI/tool steps may follow one user message.
Only AI decisions are simulated: real tools and middleware produce the results.
AI explanations are authored demonstration text, not captured private reasoning.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "circuit_breaker",
    "name": "Circuit breaker",
    "description": "Repeated forbidden filename writes stopped by native tool-call "
    "limits.",
}

# sample.py points the catalog to this module. It recognizes CONVERSATION and
# supplies it to the shared model factory, so no build_scripted_models callback is needed.
# The client extracts "client" entries. build_model(caller="worker") extracts
# "worker" entries as successive model replies. Tool and middleware entries only
# document expected results; they are never substituted for actual execution.
# There is no required alternation of user and AI: one user turn may contain
# many model/tool exchanges before the next user message.
CONVERSATION = [
    {
        "role": "client",
        "content": (
            "Reproduce the defective plan: write /slugify.py containing '# Slug implementation pending'. "
            "Repeat the identical write after rejection so the circuit breaker can stop the loop."
        ),
    },
    {
        "role": "worker",
        "content": "I will request the assigned write to /slugify.py.",
        # This proposes a write; the agent executes it after the model reply.
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/slugify.py",
                    "content": "# Slug implementation pending",
                },
                "id": "forbidden-write-1",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        # Match the result to this particular request, despite identical writes.
        "tool_call_id": "forbidden-write-1",
        "content": "Only /plan.md, /slug.py, and /test_slug.py may be written",
    },
    {
        "role": "worker",
        "content": "The backend rejected the filename. I will repeat the same write as instructed.",
        # This proposes a write; the agent executes it after the model reply.
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/slugify.py",
                    "content": "# Slug implementation pending",
                },
                "id": "forbidden-write-2",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        # Match the result to this particular request, despite identical writes.
        "tool_call_id": "forbidden-write-2",
        "content": "Only /plan.md, /slug.py, and /test_slug.py may be written",
    },
    {
        "role": "worker",
        "content": "The second write was rejected too. I will request the same write again.",
        # This proposes a write; the agent executes it after the model reply.
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/slugify.py",
                    "content": "# Slug implementation pending",
                },
                "id": "forbidden-write-3",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        # Match the result to this particular request, despite identical writes.
        "tool_call_id": "forbidden-write-3",
        "content": "Only /plan.md, /slug.py, and /test_slug.py may be written",
    },
    {
        "role": "worker",
        "content": "Three writes have been rejected. I will request a fourth identical write.",
        # This proposes a write; the agent executes it after the model reply.
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/slugify.py",
                    "content": "# Slug implementation pending",
                },
                "id": "forbidden-write-4",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        # Match the result to this particular request, despite identical writes.
        "tool_call_id": "forbidden-write-4",
        "content": "Tool call limit exceeded. Do not call 'write_file' again.",
    },
    {
        # The middleware produces this final AI message without a model call.
        # Keeping its role distinct excludes it from the worker's mock replies.
        "role": "middleware",
        "content": "'write_file' tool call limit reached: run limit exceeded (4/3 calls).",
    },
]

# Extract only user input here. The shared build_model factory independently
# extracts the worker replies into its model response collection, in list order.
USER_PROMPTS = client_prompts(CONVERSATION)
