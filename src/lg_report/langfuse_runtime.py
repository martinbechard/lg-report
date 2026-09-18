"""Shared Langfuse lifecycle for training applications; graphs remain sample-owned.

Configuration, publishing, turn grouping, and shutdown live here so samples
teach the same tracing contract without duplicating authentication plumbing.
AI attribution: Generated with AI assistance.
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context


def run_conversation(
    graph: CompiledStateGraph,
    client: Langfuse,
    callback: CallbackHandler,
    *,
    simulated: bool,
    prompts: list[str],
    trace_name: str,
) -> tuple[str, list[BaseMessage]]:
    """Invoke the ordered user prompts in one Langfuse trace and return its ID and history.

    The caller owns the configured client and must flush/shut it down, including
    on failure. The official callback captures nested graph/model operations;
    explicit turn spans preserve conversation order. Graph errors propagate and
    the Langfuse context managers mark failed spans. Inputs and outputs are sent
    to the configured Langfuse endpoint, even when the LLM is simulated.
    Traces are public so this teaching sample can be viewed without a login.
    """
    history: list[BaseMessage] = []
    # One root encompasses all invocations. Otherwise each user turn would be
    # a separate trace, obscuring the context carried into the second request.
    # Suppress ambient LangSmith tracing: this lesson has one tracing backend.
    with (
        tracing_context(enabled=False),
        client.start_as_current_observation(
            name=trace_name,
            as_type="agent",
            metadata={"sample": trace_name, "simulated": simulated},
            input=prompts,
        ) as conversation,
    ):
        # Publish this teaching trace so its link opens without a browser login.
        # The local server remains bound to localhost; this does not disable API auth.
        conversation.set_trace_as_public()
        for number, prompt in enumerate(prompts, 1):
            with client.start_as_current_observation(
                name=f"Turn {number}",
                as_type="span",
                input=prompt,
                metadata={"report_turn": number},
            ) as turn:
                # Supply the complete previous result, not just the final text.
                # This retains assistant/tool messages when a live model uses tools.
                result = graph.invoke(
                    {"messages": [*history, {"role": "user", "content": prompt}]},
                    config={
                        "callbacks": [callback],
                        "recursion_limit": 30,
                        "metadata": {
                            "report_turn": number,
                            "report_description": "Execute the user turn, including any delegated work.",
                            "simulated": simulated,
                        },
                    },
                )
                history = result["messages"]
                turn.update(output=history[-1].content)
                print(
                    f"Turn {number}\nUser: {prompt}\nAssistant: {history[-1].content}\n"
                )
        conversation.update(output=history[-1].content)
        return conversation.trace_id, history


def launch(
    *,
    app_file: str,
    description: str,
    create_graph: Callable[[bool], CompiledStateGraph],
    prompts: list[str],
    trace_name: str,
) -> None:
    """Load this sample's .env, check Langfuse access, and run the conversation.

    Default mode simulates the LLM; --live requires provider credentials. Missing
    Langfuse configuration or failed authentication stops execution before any
    model invocation. This command creates no HTML, Excel, or local trace files.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--env-file", type=Path, default=Path(app_file).with_name(".env")
    )
    parser.add_argument(
        "--live", action="store_true", help="Use the configured LLM provider"
    )
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    required = ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    values = {name: os.getenv(name, "").strip() for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        parser.error("Set " + ", ".join(missing) + " in the sample .env")
    url = urlparse(values["LANGFUSE_BASE_URL"])
    if url.scheme not in {"http", "https"} or not url.netloc:
        parser.error("LANGFUSE_BASE_URL must be an http(s) URL")

    client = Langfuse(
        public_key=values["LANGFUSE_PUBLIC_KEY"],
        secret_key=values["LANGFUSE_SECRET_KEY"],
        base_url=values["LANGFUSE_BASE_URL"],
        environment="development",
        sample_rate=1.0,
        tracing_enabled=True,
    )
    try:
        # Fail before spending model tokens if the configured project is unusable.
        if not client.auth_check():
            parser.exit(
                1,
                "Langfuse authentication failed. Check the endpoint and project keys.\n",
            )
        graph = create_graph(args.live)
        callback = CallbackHandler(public_key=values["LANGFUSE_PUBLIC_KEY"])
        trace_id, _ = run_conversation(
            graph,
            client,
            callback,
            simulated=not args.live,
            prompts=prompts,
            trace_name=trace_name,
        )
        # Short-lived programs must flush before exiting. Ingestion is asynchronous;
        # a printed URL identifies the trace, not proof the server has indexed it.
        client.flush()
        print(f"Trace: {client.get_trace_url(trace_id=trace_id)}")
    except Exception as exc:  # noqa: BLE001 - redact provider/transport exception payloads
        parser.exit(
            1,
            f"Sample failed ({type(exc).__name__}). Check configuration and Langfuse.\n",
        )
    finally:
        client.shutdown()
