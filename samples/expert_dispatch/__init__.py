"""Runnable lesson showing a dispatcher selecting among three expert agents.

Application wiring and scenario fixtures live here; reusable roles and their
composition are in agent_runtime.agents and agent_runtime.workflows respectively.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# AI attribution: Modified with AI assistance.
# This package is intentionally a thin lesson boundary: the app selects the
# runtime and the fixture scripts model decisions, while reusable dispatch and
# expert behavior remains in agent_runtime. Importing it therefore has no provider
# or retrieval side effects.
