"""Group shared runtime services used to launch and interact with agent applications.

Clients, conversation history, provider configuration, scripted models, and
tracing setup live here. Reusable clients must not contain a sample's prompts
or expected answers; those belong to its test case.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""
