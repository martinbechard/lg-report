"""Sequence authored user requests independently of console or web presentation.

A ScriptPrompter belongs to one conversation. It knows neither models nor graph
execution. Exhaustion returns None explicitly; scripts never wrap or switch to
human input silently. The frontend has its own WebScriptPrompter counterpart.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from collections.abc import Iterable

from .conversation import Request


class ScriptPrompter:
    """Keep an independent cursor over a stable copy of authored requests."""

    def __init__(self, requests: Iterable[Request]):
        self._requests = tuple(requests)
        self.position = 0

    def peek(self) -> Request | None:
        """Expose the next request without consuming it."""
        return (
            self._requests[self.position]
            if self.position < len(self._requests)
            else None
        )

    def next(self) -> Request | None:
        """Consume one request, or report exhaustion without moving the cursor."""
        request = self.peek()
        if request is not None:
            self.position += 1
        return request

    def reset(self) -> None:
        """Restart explicitly when starting a fresh conversation."""
        self.position = 0
