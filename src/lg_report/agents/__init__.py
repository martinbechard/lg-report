"""Group agent definitions independently of clients, test cases, and reporting.

Each agent belongs in its own file named after it. System instructions and graph
construction live here; user prompts arrive at runtime. Workflows select and
connect these roles; samples supply model fixtures. Importing this namespace
does not compile a graph, invoke a model, or start an external process.

AI attribution: Generated with AI assistance.

Design: docs/chat-composition.md.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""
