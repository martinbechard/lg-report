"""Provide application functions that agents can call as tools.

Workflow-reference, domain-reference, fictional service-evidence, and Wikipedia
retrieval tools supply evidence independently of clients, provider configuration,
and reporting. Importing this package executes
no tool and performs no network access. Tool functions validate their narrow
arguments; callers handle explicit misses or fixture failures.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""
