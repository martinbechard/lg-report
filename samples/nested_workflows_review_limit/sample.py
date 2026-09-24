"""Declare a teaching variant that shares another sample's implementation.

The catalog uses SAMPLE for selection and loads conversation/model factories
from the declared implementation. Importing this file does not build a model.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "nested_workflows_review_limit",
    "name": "Nested workflows: review circuit breaker",
    "description": "The coding workflow exhausts its review rounds, returns blocked, and "
    "never invokes the tester.",
    "options": {
        "scenario": "review_limit",
        "max_review_rounds": 3,
        "max_coding_cycles": 3,
    },
    "implementation": "nested_workflows",
}
