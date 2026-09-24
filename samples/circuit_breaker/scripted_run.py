"""Reproduce repeated forbidden writes without paying for a model loop.

Only model decisions are scripted. Real file tools reject the first three
requests and real middleware blocks the fourth. No scripted final response
stands in for the breaker: its termination message comes from LangChain.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from agent_runtime.harness.simulated_model import MeteredDemoModel

USER_PROMPTS = [
    ("Reproduce the defective plan: write /slugify.py containing '# Slug implementation pending'. "
     "Repeat the identical write after rejection so the circuit breaker can stop the loop.")
]


def build_models(options):
    """Supply four identical decisions with distinct tool IDs for valid pairing."""
    return {"worker": MeteredDemoModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "write_file",
            "args": {"file_path": "/slugify.py", "content": "# Slug implementation pending"},
            "id": f"forbidden-write-{attempt}",
        }])
        for attempt in range(1, 5)
    ])}
