"""Run a static or console Deep Agent conversation through Wikipedia MCP.

The server reads the prebuilt RAG index. The shared launcher owns clients,
configuration, token recording, and HTML export; the workflow bridges async MCP.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# AI attribution: Modified with AI assistance.
# The launcher leaves MCP startup and failure reporting to the shared workflow
# boundary, where this optional integration's limits remain visible.
from lg_report.platform.conversation import Request
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import launch_local
from lg_report.platform.static_client import StaticClient
from lg_report.workflows.mcp_rag_chat import build_workflow

from .test_case import USER_PROMPTS, make_simulated_model


def create_run(live):
    """Prepare a Wikipedia conversation that exercises the MCP search boundary.

    Choose a model and return a graph whose invocation performs MCP work.

    Args:
        live: Select configured provider calls when true; otherwise use the
            deterministic model fixture.

    Returns:
        The MCP workflow and the provider/model labels used in the report.

    Side effects:
        Live construction reads provider configuration. The MCP server is not
        contacted until the returned graph executes.
    """
    if live:
        model, provider, model_name = configured_model()
    else:
        model, provider, model_name = make_simulated_model(), "demo", "scripted-chat"
    return build_workflow(model), provider, model_name


def main():
    """Record the model/MCP/model exchange using standard sample options.

    The static client supplies the one repeatable raven question; console mode
    still uses the same graph. The scripted final response intentionally
    acknowledges retrieval rather than claiming to synthesize live evidence.
    """
    launch_local(
        app_file=__file__,
        description=__doc__,
        create_run=create_run,
        # The launcher calls this zero-argument factory only for static input.
        # Request holds one user turn; StaticClient supplies these turns in order.
        # Creating the client does not invoke the graph or supply model answers.
        make_static_client=lambda: StaticClient([Request(p) for p in USER_PROMPTS]),
        title="MCP RAG · Deep Agents and Wikipedia",
    )


if __name__ == "__main__":
    main()
