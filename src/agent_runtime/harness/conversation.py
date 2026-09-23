"""Run a conversation between an agent and an interchangeable user-side client.

Request and Attachment describe one submitted turn. ChatClient lets a console,
static test case, or future user-avatar produce requests and consume results.
The checkpoint owns message history; Conversation supplies client turns; the agent
never imports test prompts or needs to know which client supplied them.

Attachments are already-read text included in user context, not filesystem
access permissions. A client returning None ends the session; interrupts are answered through the client and resumed through the shared driver.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol


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
    list. ``answer`` supplies user responses to workflow interruptions without
    consuming another request. Implementations own their input and presentation;
    they need not inherit from this protocol to satisfy it.

    A client may additionally provide ``event(event)`` to observe streamed AG-UI
    events. Conversation checks for that optional hook before calling it; it is
    not required by this protocol.
    """

    def receive(self) -> Request | None:
        """Produce a user turn or signal that this client has finished."""
        ...

    def respond(self, result: dict) -> None:
        """Consume an agent result; the client owns how it is shown or evaluated."""
        ...

    def answer(self, payload: Any) -> Any:
        """Return the response used to resume one workflow interruption.

        The workflow defines both the payload and response formats, so neither
        is restricted to text. Ask the user or use a configured answer callback
        without advancing receive(). A client unable to answer must raise an
        error rather than silently approve; errors propagate to the caller.
        """
        ...


class Conversation:
    """Drive client requests through LangGraphAgent, preserving one thread.

    This is a synchronous client event pump for command-line entry points and
    test fixtures, not a graph runner. The official adapter executes every run
    and resume. The checkpoint owns history; this class never reconstructs it.
    """

    def __init__(self, graph, client: ChatClient, *, turn_scope=None):
        self.graph = graph
        self.client = client
        self.turn_scope = turn_scope

    def invoke(self, inputs, config):
        """Bridge the synchronous recorder to an async client event consumer."""
        import asyncio

        return asyncio.run(self._consume(inputs, config))

    async def _consume(self, inputs, config):
        """Submit requests and answer pauses without direct graph invocation."""
        from uuid import uuid4

        from ag_ui.core import RunAgentInput, UserMessage
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_runtime.harness.execution import (
            create_langgraph_agent,
            interaction_payload,
            open_graph,
        )

        thread = str(uuid4())
        saver = InMemorySaver()
        result = None
        turn = 0
        while (request := self.client.receive()) is not None:
            turn += 1
            scope = (
                self.turn_scope(turn, request)
                if self.turn_scope
                else nullcontext(lambda result: None)
            )
            with scope as complete:
                async with open_graph(self.graph) as graph:
                    driver = create_langgraph_agent(
                        graph,
                        checkpointer=saver,
                        config={
                            **config,
                            "metadata": {
                                **config.get("metadata", {}),
                                "report_turn": turn,
                                "thread_id": thread,
                            },
                        },
                    )
                    data = RunAgentInput(
                        thread_id=thread,
                        run_id=str(uuid4()),
                        messages=[
                            UserMessage(id=str(uuid4()), content=request.content())
                        ],
                        state=inputs if turn == 1 else {},
                        tools=[],
                        context=[],
                        forwarded_props={},
                    )
                    while True:
                        pending = []
                        async for event in driver.run(data):
                            # The consumer may display streamed events. Final
                            # state remains useful to scripted clients and recorders.
                            if hasattr(self.client, "event"):
                                self.client.event(event)
                            if event.type == "RUN_ERROR":
                                raise RuntimeError(event.message)
                            if (
                                event.type == "RUN_FINISHED"
                                and getattr(event.outcome, "type", None) == "interrupt"
                            ):
                                pending = event.outcome.interrupts
                        snapshot = await graph.aget_state(
                            {"configurable": {"thread_id": thread}}
                        )
                        result = dict(snapshot.values)
                        if not pending:
                            break
                        # Waiting is a normal interaction. Client code owns I/O;
                        # the driver owns converting these answers to graph resumes.
                        answers = [
                            {
                                "interrupt_id": item.id,
                                "status": "resolved",
                                "payload": self.client.answer(
                                    interaction_payload(item)
                                ),
                            }
                            for item in pending
                        ]
                        data = RunAgentInput(
                            thread_id=thread,
                            run_id=str(uuid4()),
                            messages=[],
                            state={},
                            tools=[],
                            context=[],
                            forwarded_props={},
                            resume=answers,
                        )
                    complete(result)
            self.client.respond(result)
            if result.get("status") == "cancelled":
                return result
        return result
