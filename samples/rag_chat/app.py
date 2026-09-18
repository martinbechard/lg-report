"""Run a static or console RAG conversation over the prebuilt Chroma index.

Ingestion is an explicit separate command. This entry point selects the chat
model and client; the agent owns retrieval and the platform owns recording.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.rag_chat import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


def create_run(live):
    """Choose the LLM; the Wikipedia agent opens its own fixed knowledge source."""
    if live:
        # Only explicit live mode can make paid chat requests; embedding stays local.
        model, provider, model_name = configured_model()
    else:
        model, provider, model_name = (
            make_simulated_model(),
            "demo",
            "scripted-chat",
        )
    return build_workflow(model), provider, model_name


def main():
    """Use the shared conversation loop so retrieval tokens appear in the report."""
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        make_static_client=lambda: StaticClient(
            [Request(prompt) for prompt in USER_PROMPTS]
        ),
        title="RAG · WikiText-103 and Chroma",
    )


if __name__ == "__main__":
    main()
