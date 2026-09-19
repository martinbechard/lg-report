"""Trace a real parent/subagent handoff with the official Langfuse callback.

The graph and fixtures are shared with the local-report sample so this lesson
changes only the tracing backend. The task tool propagates callbacks into the
specialist; attaching another callback there would risk duplicate observations.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.langfuse_runtime import launch
from lg_report.platform.static_client import StaticClient
from samples.subagent_chat.app import USER_PROMPTS, create_graph


def main() -> None:
    """Run the shared delegation graph and record its nested Langfuse trace.

    launch reads this directory's .env and CLI mode, validates tracing access,
    and flushes the callback before the CLI exits. It captures both agents under
    one root; no second callback is needed on the specialist. The --public-trace option
    exposes captured lesson content to anyone with access to the URL. This variant
    writes no local HTML; graph and provider failures are reported by the launcher.
    """
    launch(
        app_file=__file__,
        description=__doc__,
        create_graph=create_graph,
        # The launcher calls this zero-argument factory only for static input.
        # Request holds one user turn; StaticClient supplies these turns in order.
        # Creating the client does not invoke the graph or supply model answers.
        make_static_client=lambda: StaticClient(
            [Request(prompt) for prompt in USER_PROMPTS]
        ),
        trace_name="subagent-chat-langfuse",
    )


if __name__ == "__main__":
    main()
