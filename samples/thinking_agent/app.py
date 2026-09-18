"""Wire investigation_agent to a user client and local reports.

The scenario owns fixed prompts and responses. Console mode uses the same graph
with a configured provider; tracing and conversation history remain shared.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.thinking_agent import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


def create_run(live: bool):
    """Build the workflow and return its provider and model name for reporting.

    ``live`` is the explicit CLI choice: True constructs configured API clients;
    False creates fresh scripted models for this sample's fixed test case. Neither
    branch invokes an LLM here. Configuration failures propagate to the launcher
    so a requested provider run cannot quietly become a simulated success.
    """
    # Offline fixtures have fixed answers; only --live supports arbitrary user
    # questions. Both choices still execute the same workflow and tracing path.
    if live:
        model, provider, model_name = configured_model()
    else:
        model, provider, model_name = make_simulated_model(), "demo", "scripted-chat"
    return build_workflow(model), provider, model_name


def main() -> None:
    """Select a client and record one complete conversation."""
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        make_static_client=lambda: StaticClient(
            [Request(prompt) for prompt in USER_PROMPTS]
        ),
        title="Investigation · reasoning and tools",
    )


if __name__ == "__main__":
    main()
