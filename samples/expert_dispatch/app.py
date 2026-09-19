"""Launch the expert-dispatch workflow with static or console user input.

This file chooses models and clients, not routing decisions. Live questions are
routed by the dispatcher model using expert descriptions from the workflow.
See docs/chat-composition.md for component ownership.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.expert_dispatch import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


def create_run(live):
    """Prepare a dispatch lesson so questions can reach the appropriate expert.

    Share one LLM across the dispatcher and all experts.

    live=True selects the provider adapter; False selects the test simulator.
    Only the explicit live flag enables provider calls. Configuration failures
    propagate instead of silently substituting simulated answers. The returned
    identity values let the common recorder label usage without creating a
    second model or changing the dispatcher/expert graph.

    Args:
        live: Whether to construct the configured provider model.

    Returns:
        The workflow plus provider and model labels consumed by launch_local.

    Side effects:
        The live branch reads provider configuration. Model invocation is left
        to launch_local so construction itself remains easy to inspect.
    """
    if live:
        model, provider, name = configured_model()
    else:
        model = make_simulated_model()
        provider, name = "demo", "scripted-chat"
    return build_workflow(model), provider, name


def main():
    """Record one complete dispatcher session, including nested expert calls.

    launch_local owns input selection, conversation history, accounting, and
    HTML output. The static client is a repeatable teaching scenario; console
    mode can exercise the same routing graph with a human question.
    """
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        # The launcher calls this zero-argument factory only for static input.
        # Request holds one user turn; StaticClient supplies these turns in order.
        # Creating the client does not invoke the graph or supply model answers.
        make_static_client=lambda: StaticClient([Request(p) for p in USER_PROMPTS]),
        title="Dispatcher · movies, sports, and history",
    )


if __name__ == "__main__":
    main()
