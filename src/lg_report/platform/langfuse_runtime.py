"""Shared Langfuse lifecycle for training applications; callers supply the graphs.

Configuration, publishing, turn grouping, and shutdown live here so samples
teach the same tracing contract without duplicating authentication plumbing.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context

from .console_client import ConsoleClient
from .conversation import ChatClient, Conversation, Request


def run_conversation(
    graph: CompiledStateGraph,
    client: Langfuse,
    callback: CallbackHandler,
    *,
    chat_client: ChatClient,
    simulated: bool,
    trace_name: str,
    public_trace: bool = False,
) -> tuple[str, list[BaseMessage]]:
    """Observe the shared conversation loop with one root and one scope per turn.

    graph is the compiled workflow containing its agents and model adapters.
    client is the Langfuse SDK connection, while chat_client is the user-facing
    input/output channel. callback forwards graph/model events to that SDK.
    chat_client supplies requests dynamically; no prompt list is required here.
    The caller owns SDK flush/shutdown. Content is sent to the configured endpoint,
    including in simulation. Publishing requires explicit public_trace=True.
    Return trace ID and final history, or an empty history for no submitted turns.
    See docs/chat-composition.md for ownership, lifecycle, and failure semantics.
    """
    submitted_requests = []
    # Root covers the whole session; the shared loop supplies turn boundaries.
    # Disable ambient LangSmith so this example sends to one tracing backend.
    with (
        tracing_context(enabled=False),
        client.start_as_current_observation(
            name=trace_name,
            as_type="agent",
            metadata={"sample": trace_name, "simulated": simulated},
        ) as root,
    ):
        # Private is the default because console requests can include user files.
        # Explicit publication applies equally to local and hosted endpoints.
        if public_trace:
            root.set_trace_as_public()

        @contextmanager
        def turn_scope(number: int, request: Request):
            """Record actual submitted input, and let scope exit record failures."""
            submitted_requests.append(request.content())
            root.update(input=list(submitted_requests))
            with client.start_as_current_observation(
                name=f"Turn {number}",
                as_type="span",
                input=request.content(),
                metadata={"report_turn": number},
            ) as turn:

                def complete(result):
                    # Paused/empty results need no fabricated assistant answer.
                    messages = result.get("messages", [])
                    turn.update(
                        output=messages[-1].content if messages else None,
                        metadata={
                            "report_turn": number,
                            "interrupted": bool(result.get("__interrupt__")),
                        },
                    )

                yield complete

        result = Conversation(graph, chat_client, turn_scope=turn_scope).invoke(
            {},
            {
                "callbacks": [callback],
                "recursion_limit": 30,
                "metadata": {
                    "report_description": "Execute the user turn, including any delegated work.",
                    "simulated": simulated,
                },
            },
        )
        # None means the user ended before sending anything; do not index an
        # absent final message or describe an evidence-free session as successful.
        history = result.get("messages", []) if result is not None else []
        # These states describe whether the conversation completed, not whether
        # the tracing server accepted it. An approval pause is deliberately not
        # labelled success even though the graph returned without an exception.
        if result is None:
            status = "incomplete"
        elif result.get("__interrupt__"):
            status = "interrupted"
        else:
            status = "ok"
        root.update(
            output=history[-1].content if history else None,
            metadata={
                "sample": trace_name,
                "simulated": simulated,
                "session_status": status,
            },
        )
        return root.trace_id, history


def launch(
    *,
    app_file: str,
    description: str,
    create_graph: Callable[[bool], CompiledStateGraph],
    make_static_client: Callable[[], ChatClient],
    trace_name: str,
) -> None:
    """Select a user client, authenticate tracing, then run and flush one session.

    create_graph(live) constructs the executable workflow after access is
    verified; live selects real model adapters instead of scripted test adapters.
    app_file locates this application's .env, description labels CLI help, and
    trace_name identifies the session in Langfuse.
    make_static_client supplies scenario requests; console input is shared across
    samples. CLI parsing owns invalid combinations. No local report is generated.
    SDK shutdown runs on success and failure, including interactive interruption.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--env-file", type=Path, default=Path(app_file).with_name(".env")
    )
    parser.add_argument(
        "--live", action="store_true", help="Use the configured LLM provider"
    )
    parser.add_argument("--client", choices=("static", "console"), default="static")
    parser.add_argument(
        "--public-trace",
        action="store_true",
        help="Publish captured content for trace-link viewers",
    )
    args = parser.parse_args()
    # Fixed model answers cannot meaningfully answer arbitrary console requests.
    if args.client == "console" and not args.live:
        parser.error(
            "The console client requires --live; use static for offline examples."
        )
    load_dotenv(args.env_file, override=False)
    required = ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    values = {name: os.getenv(name, "").strip() for name in required}
    # Values were stripped above: absent, blank, and whitespace-only settings all
    # count as missing. Report names together without exposing credential values.
    missing = [name for name, value in values.items() if not value]
    # Any missing endpoint/key makes trace delivery impossible. Stop before
    # creating either the tracing client or a graph that could call a paid model.
    if missing:
        parser.error("Set " + ", ".join(missing) + " in the sample .env")
    endpoint_url = urlparse(values["LANGFUSE_BASE_URL"])
    # Both predicates matter: the scheme must support HTTP transport and netloc
    # must name a host. A relative path or "http:" alone is not a server endpoint.
    if endpoint_url.scheme not in {"http", "https"} or not endpoint_url.netloc:
        parser.error("LANGFUSE_BASE_URL must be an http(s) URL")

    tracing_client = Langfuse(
        public_key=values["LANGFUSE_PUBLIC_KEY"],
        secret_key=values["LANGFUSE_SECRET_KEY"],
        base_url=values["LANGFUSE_BASE_URL"],
        environment="development",
        sample_rate=1.0,
        tracing_enabled=True,
    )
    try:
        # Syntactically valid settings do not establish project access. A false
        # auth_check means these keys cannot access this endpoint; abort before
        # spending model tokens on a run whose trace could not be delivered.
        if not tracing_client.auth_check():
            parser.exit(
                1,
                "Langfuse authentication failed. Check the endpoint and project keys.\n",
            )
        graph = create_graph(args.live)
        # Each session owns a fresh client; static fixtures are consumed once.
        if args.client == "console":
            print("Console chat · /attach PATH (UTF-8 text), /send, /quit")
            chat_client = ConsoleClient()
        else:
            chat_client = make_static_client()
        # Explicit project selection keeps callback spans with this client when
        # another Langfuse client exists in the same Python process.
        callback = CallbackHandler(public_key=values["LANGFUSE_PUBLIC_KEY"])
        trace_id, _ = run_conversation(
            graph,
            tracing_client,
            callback,
            simulated=not args.live,
            chat_client=chat_client,
            public_trace=args.public_trace,
            trace_name=trace_name,
        )
        # Short-lived programs must flush before exiting. Ingestion is asynchronous;
        # a printed URL identifies the trace, not proof the server has indexed it.
        tracing_client.flush()
        print(f"Trace: {tracing_client.get_trace_url(trace_id=trace_id)}")
    except Exception as exc:  # noqa: BLE001 - redact provider/transport exception payloads
        parser.exit(
            1,
            f"Sample failed ({type(exc).__name__}). Check configuration and Langfuse.\n",
        )
    finally:
        tracing_client.shutdown()
