"""Expose the file-backed coding workflow with Langfuse session recording.

Reuse the planner, worker, test runner, reviewer, and scripted scenario from
context_budget so remote tracing observes the same work as the local report.
No models or files are created during catalog discovery.
AI attribution: Generated with OpenAI Codex assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from samples.context_budget.sample import SAMPLE as CODING_SAMPLE

# Sharing the goal keeps the adaptive user and checked-in QA rubric aligned.
SAMPLE = {
    **CODING_SAMPLE,
    "id": "context_budget_langfuse",
    "name": "Coding workflow — Langfuse",
    "description": "Build and test a Python slug function with a planner, worker, "
    "and independent reviewer; inspect their work in Langfuse.",
    "implementation": "context_budget",
    "tracing": "langfuse",
}
