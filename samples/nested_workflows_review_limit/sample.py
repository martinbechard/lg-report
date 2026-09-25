"""Demonstrate a child returning an unapproved result at its review limit.

The catalog uses the nested-workflows conversation with only one draft allowed.
The parent returns the child's unresolved feedback without claiming approval.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

SAMPLE = {
    "id": "nested_workflows_review_limit",
    "name": "Nested workflows: review limit",
    "description": "The child reaches its review limit and the parent returns the unapproved draft.",
    "options": {"max_rounds": 1},
    "implementation": "nested_workflows",
}
