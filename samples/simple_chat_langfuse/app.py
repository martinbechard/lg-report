"""Teach Langfuse tracing with a two-turn DeepAgents conversation.

This application wires the shared chat_agent to the shared conversation runtime.
It reuses the baseline chat's prompts and simulator so trace differences come from
instrumentation, not a different conversation. No local reporting callback is attached.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langgraph.graph.state import CompiledStateGraph

from lg_report.platform.conversation import Request
from lg_report.platform.langfuse_runtime import launch
from lg_report.platform.model_config import configured_model
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.simple_chat import build_workflow
from samples.simple_chat.test_case import USER_PROMPTS, make_simulated_model


# This variant changes the observation backend only. Reusing the baseline
# workflow and fixture isolates Langfuse callback behavior from graph behavior.
def create_graph(live: bool) -> CompiledStateGraph:
    """Prepare the baseline chat lesson for comparison through Langfuse traces.

    Return a fresh graph to the tracing launcher.

    live=True requires configured provider credentials and selects a real model;
    False uses a new scripted model with an empty context ledger. Missing live
    configuration raises instead of silently changing the lesson to simulation.
    This factory constructs the graph without invoking the model.
    """
    # The explicit live flag is the only switch to a provider adapter. A missing
    # credential must raise in that branch, not quietly select a scripted answer.
    return build_workflow(configured_model()[0] if live else make_simulated_model())


def main() -> None:
    """Run this directory's configured client and print its Langfuse trace URL.

    launch validates tracing credentials, attaches the official callback once,
    and flushes observations before shutdown so they are not left buffered.
    Traces are private unless --public-trace explicitly publishes their content.
    No local HTML is written; see docs/chat-composition.md for the shared design.
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
        trace_name="simple-chat-langfuse",
    )


if __name__ == "__main__":
    main()
