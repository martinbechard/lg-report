"""Define the isolated-reviewer role for one reviewed plan step.

The workflow supplies the current assignment on every invocation. Role instructions
remain outside compacted conversation history; tools enforce file ownership.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent


def build_agent(parameters: dict, skill: str):
    """Build a role without running a model; workflow code owns transitions."""
    prompt = "You are the independent read-only reviewer of ONE current task. You receive a fresh assignment, not the shared conversation. Read /plan.md and the files relevant to this task; later tasks may legitimately have no artifacts yet. Inspect the CURRENT files against the supplied task acceptance criteria. Do not edit, implement, delegate, or claim tests executed. Return only a JSON object with task_id (exact supplied ID), verdict (approve or revise), and evidence (nonempty source-backed explanation including all actionable findings and uncertainty). Approve only when this task's acceptance criteria are met. Otherwise return revise with concrete repair instructions. Approval concerns source inspection, never a passing test run." + "\n\nSample practices:\n" + skill
    agent = create_agent(**parameters, name='isolated-reviewer', system_prompt=prompt)
    agent.report_model_comment = 'Review only the assigned step'
    return agent
