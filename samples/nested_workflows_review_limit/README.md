<!-- Copyright (c) 2026 Martin.Bechard@DevConsult.ca -->
# Nested workflows: review limit

This variant runs the [nested workflow sample](../nested_workflows/README.md)
with `max_rounds=1`. In the simulated exchange, the child judge rejects the
first draft. The child returns the draft and unresolved feedback, and the parent
publishes them without claiming approval. Live model verdicts may differ.

Run `uv run python -m agent_runtime --sample nested_workflows_review_limit --demo`.
