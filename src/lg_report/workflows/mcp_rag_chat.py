"""Bridge the synchronous sample recorder to the asynchronous MCP Deep Agent.

Each turn owns an event loop and MCP connection, both closed before returning.
Conversation supplies the full history on subsequent turns. This example has no
persistent filesystem or checkpoint state across turns; evidence and messages
remain in the conversation and reporting callbacks pass through unchanged.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio

from langchain_core.language_models import BaseChatModel

from lg_report.agents.wikipedia_mcp_agent import open_agent


class MCPRagWorkflow:
    """Expose the recorder's invoke interface while MCP tools execute async."""

    def __init__(self, model: BaseChatModel):
        self.model = model

    def invoke(self, inputs, config=None):
        """Run one turn and release its MCP subprocess even when execution fails.

        This synchronous boundary is for the local CLI. Async callers should use
        the agent's open_agent context directly rather than nesting event loops.
        """

        async def run_turn():
            async with open_agent(self.model) as agent:
                return await agent.ainvoke(inputs, config=config)

        return asyncio.run(run_turn())


def build_workflow(model: BaseChatModel) -> MCPRagWorkflow:
    """Select the MCP-backed Wikipedia agent without opening storage or a model."""
    return MCPRagWorkflow(model)
