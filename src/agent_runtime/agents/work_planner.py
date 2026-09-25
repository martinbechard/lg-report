"""Turn a user request into an assignment for a worker and its reviewer.

The parent workflow owns dispatch. This agent only prepares the assignment;
it neither implements the work nor chooses review-loop transitions.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent

SYSTEM_PROMPT = (
    "Prepare one self-contained work assignment from the user's request. "
    "Preserve the requested deliverable, constraints, and acceptance criteria. "
    "Return only the assignment as plain text. Do not implement it, add unrelated "
    "requirements, or claim work has already been completed. A child workflow "
    "will write the answer and review it before returning."
)


def build_agent(parameters: dict):
    """Build the planning role using the workflow's model and runtime settings."""
    return create_agent(**parameters, name="work_planner", system_prompt=SYSTEM_PROMPT)
