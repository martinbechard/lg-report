"""Demonstrate Deep Agents using Wikipedia retrieval through a local MCP server.

The sample reuses the completed RAG index and the shared local report pipeline.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# AI attribution: Modified with AI assistance.
# The local MCP server and prebuilt index are integration dependencies, not
# import-time resources. This keeps package discovery safe when the optional
# server is unavailable and makes the app's explicit launch step meaningful.
