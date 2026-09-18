"""Launch the review-loop lesson with static or console user input.

One configured LLM serves author and judge. The high-level first-draft option is
intentional for this training sample; the reusable workflow defaults it off.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.review_loop import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


def create_run(live):
    """Return (workflow, provider, model_name) for the shared sample launcher.

    live is the CLI choice, not a connectivity check: False uses a deterministic
    test fixture; True loads the configured provider. The identity strings label
    usage in reports; they are not additional model objects.
    """
    # Only explicit live mode spends provider tokens; configuration failures propagate.
    if live:
        model, provider, name = configured_model()
    else:
        model, provider, name = make_simulated_model(), "demo", "scripted-chat"
    # Three is the maximum total drafts, not three retries after the first draft.
    # The overview flag creates a useful teaching situation in both offline and
    # live runs. Only the judge decides whether that first answer needs revision.
    return (
        build_workflow(model, max_rounds=3, first_draft_high_level=True),
        provider,
        name,
    )


def main():
    """Record every draft and judge call through the common conversation boundary."""
    # The launcher selects static/console input, retains the user conversation,
    # attaches recording, and prints the report path. This file owns the scenario
    # choices; the agent and workflow modules never import USER_PROMPTS.
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        # A fresh client consumes the scenario once per run. Human console input
        # replaces this client without changing the review graph or its budget.
        make_static_client=lambda: StaticClient(
            [Request(prompt) for prompt in USER_PROMPTS]
        ),
        title="Review loop · overview, feedback, drill-down",
    )


if __name__ == "__main__":
    main()
