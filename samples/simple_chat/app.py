"""Wire the simple-chat workflow to a client and local reporting.

User prompts and offline answers belong to test_case. The shared platform owns
console input, history, configuration, and recording lifecycle.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.simple_chat import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


# This adapter is intentionally small: it exposes the shared runtime's client,
# model-selection, and recording seams so the lesson can be compared directly
# with the tool, review, and delegation samples.
def create_run(live: bool):
    """Prepare a direct-answer conversation for the shared sample launcher.

    Return the workflow and its provider/model labels for report accounting.

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
    """Run the selected client through the same workflow and recording path."""
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        # The launcher calls this zero-argument factory only for static input.
        # Request holds one user turn; StaticClient supplies these turns in order.
        # Creating the client does not invoke the graph or supply model answers.
        make_static_client=lambda: StaticClient(
            [Request(prompt) for prompt in USER_PROMPTS]
        ),
        title="Simple chat",
    )


if __name__ == "__main__":
    main()
