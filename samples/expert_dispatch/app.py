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
    """Configure one LLM and share it across the dispatcher and all experts.

    live=True selects the provider adapter; False selects the test simulator.
    Only the explicit live flag enables provider calls. Configuration failures
    propagate instead of silently substituting simulated answers.
    """
    if live:
        model, provider, name = configured_model()
    else:
        model = make_simulated_model()
        provider, name = "demo", "scripted-chat"
    return build_workflow(model), provider, name


def main():
    """Record the entire session, including nested expert calls, exactly once."""
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        make_static_client=lambda: StaticClient([Request(p) for p in USER_PROMPTS]),
        title="Dispatcher · movies, sports, and history",
    )


if __name__ == "__main__":
    main()
