"""Run a static or console Deep Agent conversation through Wikipedia MCP.

The server reads the prebuilt RAG index. The shared launcher owns clients,
configuration, token recording, and HTML export; the workflow bridges async MCP.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.mcp_rag_chat import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


def create_run(live):
    """Choose a provider only in explicit live mode; otherwise script the LLM."""
    if live:
        model, provider, model_name = configured_model()
    else:
        model, provider, model_name = make_simulated_model(), "demo", "scripted-chat"
    return build_workflow(model), provider, model_name


def main():
    """Record the model/tool/model exchange using the standard sample options."""
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        make_static_client=lambda: StaticClient([Request(p) for p in USER_PROMPTS]),
        title="MCP RAG · Deep Agents and Wikipedia",
    )


if __name__ == "__main__":
    main()
