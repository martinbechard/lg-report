"""Correct rejected gateway decisions within the graph's model-call boundary.

A malformed response is paid inference but has not executed any application
operation. Ask for one corrected envelope at most twice. Keep each model call
in the normal trace, preserve completed tool observations, and never retry a
transport failure or turn a rejected decision into an inferred tool action.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from .gateway_model import GatewayResponseError


class GatewayDecisionMiddleware(AgentMiddleware):
    """Bound format recovery to three separately metered attempts per decision."""

    @staticmethod
    def correction(request, error):
        """Retain the rejected text as evidence, with explicit non-execution feedback."""
        return request.override(messages=[*request.messages, error.result.generations[0].message,
            HumanMessage(
                'The gateway rejected your last response. NONE of its requested operations '
                'executed. Correct the decision using the existing observations. Return exactly '
                'ONE JSON object with content (string) and tool_calls (array). Do not escape '
                'the entire object, concatenate objects, add commentary, or emit special role '
                'tokens. Request required file operations in tool_calls and STOP to await '
                'their actual results. Put any final task JSON inside the content string '
                'with an empty tool_calls array. Do not repeat previously completed operations.'
            )])

    def wrap_model_call(self, request, handler):
        """Retry only rejected decision formats; all other failures propagate."""
        for attempt in range(3):
            try:
                return handler(request)
            except GatewayResponseError as error:
                if attempt == 2:
                    raise
                request = self.correction(request, error)

    async def awrap_model_call(self, request, handler):
        """Keep asynchronous execution and correction limits identical."""
        for attempt in range(3):
            try:
                return await handler(request)
            except GatewayResponseError as error:
                if attempt == 2:
                    raise
                request = self.correction(request, error)
