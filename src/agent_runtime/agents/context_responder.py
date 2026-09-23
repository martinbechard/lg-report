"""Answer from the shared peer history and an isolated specialist's evidence.

The workflow supplies task-tool registration and context policy. This role
owns delegation instructions, not the specialist's message history or budget.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent

SYSTEM_PROMPT = (
    "You are the responding peer. Read the user's request and the planning brief. "
    "Delegate an echo demonstration to isolated-subagent using task with "
    "a self-contained assignment. Then answer concisely using the returned summary. "
    "Preserve user constraints from the shared conversation or its summary."
)


def build_agent(parameters: dict):
    """Expose only the workflow-registered task tool and explicit middleware."""
    return create_agent(
        **parameters,
        name="context_responder",
        system_prompt=SYSTEM_PROMPT,
    )
