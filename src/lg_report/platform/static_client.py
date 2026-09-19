"""Supply requests from a caller-owned static test case.

Scripted user prompts and scripted model responses are different fixtures. This
client supplies requests; a separate scripted model supplies offline answers.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from .conversation import Request


class StaticClient:
    """Supply a finite test conversation and retain each result for inspection.

    For example, StaticClient([Request("Explain ReAct")]) submits one turn;
    results then contains the complete workflow result, including its messages.
    Pass custom Requests to exercise file context or another test case. A client
    instance is consumed once; create a new one for each session.
    """

    def __init__(self, requests):
        """Prepare a reproducible user conversation from caller-supplied requests.

        ``requests`` may be any iterable; it is converted to an iterator so the
        client cannot recycle prompts after exhaustion. Results are retained in
        submission order for assertions and later report inspection.
        """
        self.requests = iter(requests)
        self.results = []

    def receive(self) -> Request | None:
        """Advance the test conversation; return its next Request or None to end it."""
        return next(self.requests, None)

    def respond(self, result: dict) -> None:
        """Save the full result so a test can assert history, tools, or output.

        The result is stored by reference, matching the graph client handoff and
        preserving interrupt/status fields for assertions.
        """
        self.results.append(result)
