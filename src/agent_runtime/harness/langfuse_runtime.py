"""Shared Langfuse lifecycle for training applications selected by package.

Configuration, publishing, turn grouping, and shutdown live here so samples
teach the same tracing contract without duplicating authentication plumbing.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from contextlib import contextmanager

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


class ConversationTrace:
    """Supply tracing hooks to a conversation without executing or recording it locally."""

    def __init__(self, client, root, name, simulated):
        """Retain the open root observation and inputs belonging to this session."""
        self.client = client
        self.root = root
        self.name = name
        self.simulated = simulated
        self.requests = []
        self.trace_id = root.trace_id

    @contextmanager
    def turn_scope(self, number, request):
        """Enclose a user turn, including resumes, in one child observation."""
        self.requests.append(request.content())
        self.root.update(input=list(self.requests))
        with self.client.start_as_current_observation(
            name=f"Turn {number}",
            as_type="span",
            input=request.content(),
            metadata={"report_turn": number},
        ) as turn:

            def complete(result):
                """Attach the completed graph result without executing another turn."""
                messages = result.get("messages", [])
                turn.update(
                    output=messages[-1].content if messages else None,
                    metadata={
                        "report_turn": number,
                        "interrupted": bool(result.get("__interrupt__")),
                    },
                )

            # Exceptions unwind through the SDK scope and mark the observation
            # as failed; the conversation only calls complete on normal return.
            yield complete

    def complete(self, result):
        """Label the session outcome; an empty conversation is not a success."""
        history = result.get("messages", []) if result is not None else []
        status = (
            "incomplete"
            if result is None
            else "interrupted"
            if result.get("__interrupt__")
            else "ok"
        )
        self.root.update(
            output=history[-1].content if history else None,
            metadata={
                "sample": self.name,
                "simulated": self.simulated,
                "session_status": status,
            },
        )


@contextmanager
def conversation_trace(client, *, trace_name, simulated, public_trace=False):
    """Open optional remote instrumentation around the caller's single execution.

    This scope owns observations only. The caller owns graph construction, local
    reporting, and SDK shutdown. Publication requires an explicit opt-in because
    requests can contain user files. Exceptions reach the SDK's failure handling.
    """
    with client.start_as_current_observation(
        name=trace_name,
        as_type="agent",
        metadata={"sample": trace_name, "simulated": simulated},
    ) as root:
        if public_trace:
            root.set_trace_as_public()
        yield ConversationTrace(client, root, trace_name, simulated)


class LangfuseCapture:
    """Own tracing resources for console or browser sessions, without executing graphs."""

    def __init__(self, *, settings=None):
        # Selecting a tracing sample is explicit. Validate credentials before
        # constructing/running its model; never silently downgrade its recorder.
        keys = ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        settings = {**(settings or {}), **os.environ}
        values = {key: settings.get(key, "").strip() for key in keys}
        if not all(values.values()):
            raise ValueError(
                "Configure LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
            )
        self.client = Langfuse(
            base_url=values[keys[0]],
            public_key=values[keys[1]],
            secret_key=values[keys[2]],
        )
        try:
            if not self.client.auth_check():
                raise ValueError(
                    "Langfuse authentication failed. Check the endpoint and project keys."
                )
            self.callback = CallbackHandler(public_key=values[keys[1]])
        except BaseException:
            self.client.shutdown()
            raise

    @contextmanager
    def scope(self, thread_id, run_id, name):
        """Group independently streamed runs by stable session trace identity."""
        with self.client.start_as_current_observation(
            name=name,
            as_type="agent",
            trace_context={"trace_id": self.client.create_trace_id(seed=thread_id)},
            metadata={"thread_id": thread_id, "run_id": run_id},
        ):
            yield
        self.client.flush()

    def close(self):
        """Flush and release exporter resources when the session is discarded."""
        self.client.shutdown()
