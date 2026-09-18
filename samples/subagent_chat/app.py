"""Wire the delegating parent and specialist to clients and local reporting.

The parent and specialist may use the same provider model. Their graphs own
separate conversation histories; offline model instances additionally need
separate answer cursors and token ledgers. Agent definitions live in lg_report.agents.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.subagent_chat import build_workflow

from .test_case import USER_PROMPTS, make_simulated_models


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
        parent_model, provider, model_name = configured_model()
        specialist_model = configured_model()[0]
    else:
        parent_model, specialist_model = make_simulated_models()
        provider, model_name = "demo", "scripted-chat"
    return build_workflow(parent_model, specialist_model), provider, model_name


def create_graph(live: bool):
    """The Langfuse application needs the same graph without local cost settings."""
    return create_run(live)[0]


def main() -> None:
    """Record the complete parent and child conversation with one callback."""
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        make_static_client=lambda: StaticClient(
            [Request(prompt) for prompt in USER_PROMPTS]
        ),
        title="Parent and specialist · delegation",
    )


if __name__ == "__main__":
    main()
