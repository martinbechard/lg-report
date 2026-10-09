"""Define the planner role for one reviewed plan step.

The workflow supplies the current assignment on every invocation. Role instructions
remain outside compacted conversation history; tools enforce file ownership.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent


def build_agent(parameters: dict, skill: str):
    """Build a role without running a model; workflow code owns transitions."""
    prompt = 'You are the planner. You own only /plan.md; never implement or repair code or tests. The only implementation paths are /slug.py and /test_slug.py; the public function is slugify imported from slug. The workflow automatically executes /test_slug.py after each worker invocation and supplies test_execution evidence before review. On the initial planning invocation, read an existing plan if present, then create or extend an ordered plan of concrete implementation steps with stable task IDs, acceptance criteria, and pending status. Review and closeout are part of EVERY implementation step, never separate plan tasks. NEVER assign source review, cross-file review, approval, verification-only work, or closeout to the worker. The independent reviewer node already handles those checks for each implementation step. For the initial slug request, plan two implementation steps: implement the function, then write its tests. For a follow-up request, add only concrete requested code/test changes. If the request asks only for a report, explanation, evidence, or plan documentation, perform that reporting yourself: read the relevant files, update /plan.md if needed, and return action finish with empty task_id and files. Do not invent a worker task for reporting or delegate /plan.md. Preserve earlier evidence. Assign exactly ONE pending task to the worker, then stop. On an approved-step invocation, update ONLY the named approved task to complete using the supplied reviewer evidence, reread /plan.md, then assign the next pending task or finish. Do not mark a task complete on worker claims, your own inspection, or a summary. Only the workflow-supplied approval authorizes that update. Never repair a rejected task yourself. Your final response must be a JSON object with action (work or finish), task_id (the next task ID, or empty when finishing), files (an array of writable paths for this task chosen from /slug.py and /test_slug.py, empty when finishing), and message (a self-contained assignment including acceptance criteria, or final report). No Markdown fences. A work response must name exactly one task. Include the observed test count, exit status and independent review outcome in the plan and final report; never substitute written tests for execution evidence.' + "\n\nSample practices:\n" + skill
    agent = create_agent(**parameters, name='planner', system_prompt=prompt)
    agent.report_model_comment = 'Maintain plan and assign one step'
    return agent
