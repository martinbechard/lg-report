<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Nested workflows: test-fix circuit breaker

[sample.py](sample.py) registers this variant with its own options and
configuration directory. The shared [`nested_workflows` implementation](../nested_workflows/README.md)
provides its workflow and chronological script.

Run the demonstration with `uv run python -m agent_runtime --sample nested_workflows_test_limit --demo`.
