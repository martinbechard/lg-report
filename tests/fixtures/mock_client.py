"""Capture conversation results in tests without terminal I/O.

Scripted user prompts and scripted model responses are different fixtures. This
client supplies requests; a separate scripted model supplies offline answers.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.conversation import Request
from agent_runtime.harness.script_prompter import ScriptPrompter


class MockClient:
    """Supply a finite test conversation and retain each result for inspection.

    For example, MockClient([Request("Explain ReAct")]) submits one turn;
    results then contains the complete workflow result, including its messages.
    Pass custom Requests to exercise file context or another test case. A client
    instance is consumed once; create a new one for each session.
    """

    def __init__(self, requests, *, answer=None):
        """Prepare a reproducible user conversation from caller-supplied requests.

        ``requests`` may be any iterable; ScriptPrompter keeps an independent
        position and reports exhaustion without recycling prompts. Results are retained in
        submission order for assertions and later report inspection.

        ``answer`` is an optional callback for user-side responses to workflow
        interruptions, such as file-change approval or a clarification question.
        It receives the workflow's interaction payload and returns the response
        used to resume that interruption. It does not supply AI model responses
        or consume the next item from ``requests``. The workflow determines the
        expected response format; the callback may return a scripted decision
        or ask a person, for example by using ``ConsoleClient().answer``.

        Construction only stores the callback. When an interruption occurs,
        Conversation calls this client's ``answer(payload)`` method, which then
        invokes it. Omit the callback for conversations without interruptions.
        If an interruption occurs with ``answer=None``, the method raises
        ValueError rather than inventing an answer or automatically approving.
        """
        self.answer_callback = answer
        self.prompter = ScriptPrompter(requests)
        self.results = []

    def receive(self) -> Request | None:
        """Advance the test conversation; return its next Request or None to end it."""
        return self.prompter.next()

    def respond(self, result: dict) -> None:
        """Save the full result so a test can assert history, tools, or output.

        The result is stored by reference, matching the graph client handoff and
        preserving interrupt/status fields for assertions.
        """
        self.results.append(result)

    def answer(self, payload):
        """Obtain a user response to one pending workflow interruption.

        ``payload`` is the workflow-authored question or approval details, passed
        unchanged to the callback supplied at construction. Return its result
        unchanged so Conversation can submit it as the interruption's resume
        payload. Missing callbacks raise ValueError; callback errors propagate.
        """
        if self.answer_callback is None:
            raise ValueError(
                "This scripted client has no response for the pending interaction"
            )
        return self.answer_callback(payload)
