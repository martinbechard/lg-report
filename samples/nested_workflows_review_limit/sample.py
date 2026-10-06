"""Demonstrate a child returning an unapproved result at its review limit.

The catalog uses the nested-workflows conversation with only one draft allowed.
The parent returns the child's unresolved feedback without claiming approval.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

SAMPLE = {
    # The adaptive test user evaluates this goal before requesting another turn.
    "goal": 'The parent returns the child result at its configured review limit and clearly discloses the actual approval status and unresolved review feedback. An unapproved draft must not be presented as approved.',
    "id": "nested_workflows_review_limit",
    "name": "Nested workflows: review limit",
    "description": "The child reaches its review limit and the parent returns the unapproved draft.",
    "options": {"max_rounds": 1},
    "implementation": "nested_workflows",
}
