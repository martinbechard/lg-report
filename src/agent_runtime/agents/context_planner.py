"""Read the shared conversation and prepare a brief for the next peer agent.

The workflow owns context retention. This role owns only planning instructions;
it does not summarize or decide how much conversation another agent receives.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent

SYSTEM_PROMPT = (
    "You are the planning peer. Read the latest user request and earlier context. "
    "Write a short brief for the responding peer, preserving user constraints. "
    "Do not answer the user or call tools."
)


def build_agent(parameters: dict):
    """Build a planner with the workflow's model and context middleware."""
    return create_agent(
        **parameters, name="context_planner", system_prompt=SYSTEM_PROMPT
    )
