<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Nested workflows: review circuit breaker

[sample.py](sample.py) registers this variant with its own options and
configuration directory. The shared [`nested_workflows` implementation](../nested_workflows/README.md)
provides its workflow and chronological script.

Run the demonstration with `uv run python -m agent_runtime --sample nested_workflows_review_limit --demo`.
