"""Shared Langfuse lifecycle for training applications selected by package.

Configuration, publishing, turn grouping, and shutdown live here so samples
teach the same tracing contract without duplicating authentication plumbing.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from contextlib import contextmanager
from uuid import uuid4

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler


class ConversationTrace:
    """Group independent turn traces without owning conversation state or accounting."""

    def __init__(self, client, name, simulated, session_id, public_trace):
        """Retain correlation and visibility policy; observations start with turns."""
        self.client = client
        self.name = name
        self.simulated = simulated
        self.session_id = session_id
        self.public_trace = public_trace
        self.trace_id = None

    @contextmanager
    def turn_scope(self, number, request):
        """Record one request and its resumes as a trace in the conversation session.

        A root per turn gives replay the request/answer pairs. The Turn span
        retains model/tool ancestry. SDK scopes mark exceptions as errors.
        """
        with self.client.start_as_current_observation(
            name=self.name,
            as_type="agent",
            input=request.content(),
            metadata={
                "sample": self.name,
                "simulated": self.simulated,
                "report_turn": number,
            },
        ) as root:
            self.trace_id = root.trace_id
            if self.public_trace:
                root.set_trace_as_public()
            with self.client.start_as_current_observation(
                name=f"Turn {number}",
                as_type="span",
                input=request.content(),
                metadata={"report_turn": number},
            ) as turn:

                def complete(result):
                    """Publish the final response without replaying execution."""
                    messages = result.get("messages", [])
                    output = messages[-1].content if messages else None
                    interrupted = bool(result.get("__interrupt__"))
                    turn.update(
                        output=output,
                        metadata={"report_turn": number, "interrupted": interrupted},
                    )
                    root.update(
                        output=output,
                        metadata={
                            "session_status": "interrupted" if interrupted else "ok"
                        },
                    )

                yield complete

    def session_url(self):
        """Discover the project URL through the SDK instead of hardcoding its ID."""
        if self.trace_id is None:
            return None
        trace_url = self.client.get_trace_url(trace_id=self.trace_id)
        return trace_url.rsplit("/traces/", 1)[0] + "/sessions/" + self.session_id


@contextmanager
def conversation_trace(
    client, *, trace_name, simulated, public_trace=False, session_id=None
):
    """Propagate session identity across turns and model-driven client calls.

    No conversation-wide root stays open: each turn becomes its own trace.
    UUIDs avoid embedding prompts or identifying user content in session IDs.
    Publication remains opt-in per trace; the session itself stays private.
    """
    session_id = session_id or str(uuid4())
    with propagate_attributes(session_id=session_id):
        yield ConversationTrace(client, trace_name, simulated, session_id, public_trace)


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
        """Give each browser request a trace; approval resumes share the session.

        Propagation covers the official callback and delegated tool calls.
        """
        with (
            propagate_attributes(session_id=thread_id),
            self.client.start_as_current_observation(
                name=name,
                as_type="agent",
                trace_context={
                    "trace_id": self.client.create_trace_id(
                        seed=f"{thread_id}:{run_id}"
                    )
                },
                metadata={"thread_id": thread_id, "run_id": run_id},
            ) as root,
        ):
            yield root
        self.client.flush()

    def close(self):
        """Flush and release exporter resources when the session is discarded."""
        self.client.shutdown()
