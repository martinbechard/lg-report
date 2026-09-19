"""Run a conversation between an agent and an interchangeable user-side client.

Request and Attachment describe one submitted turn. ChatClient lets a console,
static test case, or future user-avatar produce requests and consume results.
Conversation owns the complete message history and turn numbering; the agent
never imports test prompts or needs to know which client supplied them.

Attachments are already-read text included in user context, not filesystem
access permissions. A client returning None ends the session; an agent interrupt
stops the loop rather than accidentally skipping a human approval boundary.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Attachment:
    """An explicitly supplied UTF-8 text file, already read by the client.

    ``name`` is display/context metadata and ``content`` is the captured text.
    This value does not grant the graph permission to reopen the named path.
    """

    name: str
    content: str


@dataclass(frozen=True)
class Request:
    """One user turn; attachments become input context, not filesystem permissions.

    ``prompt`` may be empty when a client intentionally submits attachments
    alone. ``files`` is an immutable ordered tuple so clients cannot change a
    request after tracing or graph invocation begins.
    """

    prompt: str
    files: tuple[Attachment, ...] = ()

    def content(self) -> str:
        """Keep the prompt and named file contents together in one user message.

        Text is sent inline so both providers use the same input representation.
        A filename labels user data; it never tells the agent to open a local path.
        """
        parts = [self.prompt]
        for attachment in self.files:
            parts.append(f"Attached file: {attachment.name}\n{attachment.content}")
        return "\n\n".join(parts)


class ChatClient(Protocol):
    """User-side boundary shared by console, static test, or a future avatar agent.

    ``receive`` returns the next ``Request``, or ``None`` to end the conversation.
    ``respond`` receives the full graph result before the next receive, allowing
    an avatar to choose its next prompt from the answer instead of a predetermined
    list. Implementations should preserve their own fixture/input ownership.
    """

    def receive(self) -> Request | None:
        """Produce a user turn or signal that this client has finished."""
        ...

    def respond(self, result: dict) -> None:
        """Consume an agent result; the client owns how it is shown or evaluated."""
        ...


class Conversation:
    """Own one session's history; neither the client nor agent needs reporting code.

    Conversation(graph, client) adapts the turn loop to record_run's invoke API.
    The same graph and complete history are reused so tool exchanges and file
    contents remain in subsequent context. A new invoke starts a fresh session.
    Optional turn_scope(number, Request) returns a context manager yielding a
    completion callable for the graph result. Its exit observes exceptions; it
    must not swallow them. This lifecycle hook keeps tracing SDKs out of the loop.
    See docs/chat-composition.md for the component and sequence diagrams.
    """

    def __init__(self, graph, client: ChatClient, *, turn_scope=None):
        """Keep the executable workflow and user client separate from model adapters.

        graph is an already-built message-based workflow, not an LLM object.
        client owns input/output; turn_scope optionally records each invocation.
        Neither construction nor this assignment calls the model or reads input.
        """
        self.graph = graph
        self.client = client
        self.turn_scope = turn_scope

    def invoke(self, inputs, config):
        """Let a client conduct a complete conversation with retained agent context.

        ``inputs`` is unused; requests arrive through the client. ``config`` is
        copied on every turn, preserving callbacks and other recorder settings
        while adding the one-based ``report_turn`` metadata. Graph and scope
        errors propagate to the recorder so partial traces can remain diagnostic.
        The return is the final graph result, or ``None`` when no request was
        submitted; an interrupt result ends this client because resumption is an
        explicit client responsibility.
        """
        history = []
        result = None
        turn = 0
        while True:
            request = self.client.receive()
            # None ends the session; an empty prompt with attachments is still a
            # valid request and must not be mistaken for this explicit sentinel.
            if request is None:
                return result
            turn += 1
            # Recording hooks wrap execution, not client input: typing time belongs
            # to the session, not this turn. No hook means ordinary local execution.
            scope = (
                self.turn_scope(turn, request)
                if self.turn_scope is not None
                # The no-op callback accepts the complete graph result, just
                # like a tracing hook, and intentionally records nothing.
                else nullcontext(lambda result: None)
            )
            with scope as complete:
                # LangChain accepts these role/content dictionaries as user
                # messages. invoke executes the entire agent graph, potentially
                # including model requests and tools, and returns a state mapping.
                # Tool outputs live inside its message history; result itself is
                # the outcome of the graph invocation, not one tool's output.
                result = self.graph.invoke(
                    {
                        "messages": [
                            *history,
                            {"role": "user", "content": request.content()},
                        ]
                    },
                    config={
                        **config,
                        "metadata": {**config.get("metadata", {}), "report_turn": turn},
                    },
                )
                complete(result)
            # The workflow returns the complete history, including tool exchanges.
            # Replacing our history avoids duplicating messages by appending a
            # complete transcript to an existing transcript on every turn.
            history = result["messages"]
            self.client.respond(result)
            # An interrupt is an approval/resume boundary, not permission to send
            # the next user prompt. This simple client does not implement resume.
            if result.get("__interrupt__"):
                return result
