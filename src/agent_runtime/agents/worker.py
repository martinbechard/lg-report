"""Define the worker role for one reviewed plan step.

The workflow supplies the current assignment on every invocation. Role instructions
remain outside compacted conversation history; tools enforce file ownership.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent


def build_agent(parameters: dict, skill: str):
    """Build a role without running a model; workflow code owns transitions."""
    prompt = "You are the worker. Implement or repair ONLY the current task supplied by the workflow. Read /plan.md and relevant existing files first. You may write only /slug.py and /test_slug.py, never /plan.md. The function is slugify and tests import from slug. If review feedback is supplied, fix those findings for this same task; do not advance to another task. Check tool outcomes and return concrete evidence of your changes. Do not claim approval or task completion, and do not perform your own independent review. Even if an assignment asks you for an independent review or an approval verdict, do not provide one: report your file observations as an implementation handoff and explicitly leave the approval decision to the independent reviewer. Never label your response approve, approved, or independent review. The workflow sends your work to a fresh read-only reviewer after each invocation. Tests cannot be executed by these file tools; report that limitation. Stop after this task's changes and handoff." + "\n\nSample practices:\n" + skill
    agent = create_agent(**parameters, name='worker', system_prompt=prompt)
    agent.report_model_comment = 'Implement or repair only the assigned step'
    return agent
