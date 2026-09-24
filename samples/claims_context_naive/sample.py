"""Declare a teaching variant that shares another sample's implementation.

The catalog uses SAMPLE for selection and loads conversation/model factories
from the declared implementation. Importing this file does not build a model.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "claims_context_naive",
    "name": "Claims context — naive",
    "description": "Retain earlier claim context after edits.",
    "options": {"mode": "naive"},
    "implementation": "claims_context",
}
