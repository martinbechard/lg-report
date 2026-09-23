"""Bridge the synchronous sample recorder to the asynchronous MCP Deep Agent.

Each turn owns an event loop and MCP connection, both closed before returning.
Conversation supplies the full history on subsequent turns. This example has no
persistent filesystem or checkpoint state across turns; evidence and messages
remain in the conversation and reporting callbacks pass through unchanged.

AI attribution: Modified with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio

from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel

from agent_runtime.agents.wikipedia_mcp_agent import open_agent
from agent_runtime.harness.model_factory import build_model


class MCPRagWorkflow:
    """Expose a synchronous recorder boundary around an async MCP agent.

    The wrapper owns one model reference but deliberately does not retain an MCP
    client or event loop between calls.  This keeps subprocess and connection
    lifetime tied to one turn, which is appropriate for the local synchronous
    sample but means callers should use ``open_agent`` directly for long-lived
    async sessions.
    """

    def __init__(self, model: BaseChatModel):
        """Prepare a reusable workflow adapter for synchronous Wikipedia conversations.

        ``model`` is an already configured live or simulated chat model.  The
        constructor makes no provider request and opens no subprocess; those
        effects belong to ``invoke`` so a discarded workflow is inert.
        """
        # Retain only the model dependency. Transport commands, discovery, tool
        # registration, and role instructions stay behind open_agent's interface.
        self.parameters = {"model": model, "backend": StateBackend()}

    def invoke(self, inputs, config=None):
        """Answer a conversation turn using the MCP-backed Wikipedia agent.

        Return the whole agent's final state mapping, including its ``messages``
        history. Individual executed searches produce tool messages within that
        history; the returned mapping is not one search tool's result.

        ``inputs`` and ``config`` are passed through to the agent unchanged, so
        the caller remains responsible for supplying conversation history and
        tracing callbacks.  The async context manager closes the MCP connection
        on success or error.  This synchronous boundary is for the local CLI;
        async callers should use the agent's open_agent context directly rather
        than nesting event loops.
        """

        async def run_turn():
            """Keep this turn's agent alive until it has finished using MCP tools.

            Capture ``inputs`` (conversation state) and ``config`` (execution
            options/callbacks) from invoke. Return the final state from ainvoke;
            leaving the context closes MCP before the synchronous caller resumes.
            """
            # Entering opens the subprocess and discovers its tool schemas.
            # ainvoke executes the graph, including any model-selected searches;
            # awaiting it waits for the entire turn, not just one tool response.
            async with open_agent(self.parameters) as agent:
                return await agent.ainvoke(inputs, config=config)

        # asyncio.run owns a fresh loop and blocks until run_turn returns or
        # raises. It cannot be nested in an already running loop on this thread.
        return asyncio.run(run_turn())


def build_workflow(model: BaseChatModel | None = None) -> MCPRagWorkflow:
    """Give synchronous application code the MCP Wikipedia conversation interface.

    ``model`` is the configured live or scripted chat model used on every turn.

    Workflow construction is intentionally side-effect free.  The returned
    wrapper creates and closes the per-turn MCP context only when invoked.
    """
    # Model selection belongs to the factory; the workflow supplies its role.
    if model is None:
        model = build_model(caller="workflow")
    # This adapter is a lifecycle harness, not another agent. It translates the
    # recorder's synchronous invocation into one scoped async agent session.
    return MCPRagWorkflow(model)
