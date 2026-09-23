"""Apply workflow-selected history compaction and an estimated input ceiling.

One immutable budget can configure multiple peers, but their workflow must also
pass shared messages. A child gets a separate budget and separate middleware.
Counts are local estimates, not provider tokenizer guarantees. The ceiling
covers agent requests, including instructions/tools; summary-model calls have
their own provider limits. Output allowance must be reserved by the caller.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from dataclasses import dataclass
from math import ceil

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool


class ContextBudgetExceeded(ValueError):
    """Stop before an oversized agent request instead of silently losing content."""


class InputBudgetMiddleware(AgentMiddleware):
    """Check the complete agent request after history summarization has run."""

    def __init__(self, limit: int):
        self.limit = limit

    def check(self, request):
        """Include role instructions and tool schemas in the local size estimate.

        The latest message or an indivisible tool exchange may exceed the budget
        even after summarization. Reject it visibly; do not truncate user intent
        or send a request that violates the declared local estimate.
        """
        messages = list(request.messages)
        if request.system_message is not None:
            messages.insert(0, request.system_message)
        tokens = count_tokens_approximately(messages)
        if request.tools:
            schemas = [convert_to_openai_tool(tool) for tool in request.tools]
            tokens += ceil(len(json.dumps(schemas, ensure_ascii=False)) / 4)
        if tokens > self.limit:
            raise ContextBudgetExceeded(
                f"Estimated agent input {tokens} tokens exceeds budget {self.limit}; "
                "shorten the input or raise this workflow/subagent budget."
            )

    def wrap_model_call(self, request, handler):
        """Guard synchronous calls after before-model compaction updates state."""
        self.check(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        """Apply the identical guard to browser/AG-UI asynchronous execution."""
        self.check(request)
        return await handler(request)


@dataclass(frozen=True)
class ContextBudget:
    """Configure one conversation's policy, independently of model capacity.

    trigger_tokens counts history before compaction; keep_tokens targets recent
    unsummarized history. Whole messages/tool pairs can exceed that target.
    max_input_tokens checks the estimated final request including its envelope.
    """

    max_input_tokens: int
    trigger_tokens: int
    keep_tokens: int

    def __post_init__(self):
        """Reject inconsistent budgets before constructing or invoking models."""
        values = (self.max_input_tokens, self.trigger_tokens, self.keep_tokens)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Context budget values must be positive integers")
        if not self.keep_tokens < self.trigger_tokens < self.max_input_tokens:
            raise ValueError("Require keep_tokens < trigger_tokens < max_input_tokens")

    def middleware(self, summary_model: BaseChatModel) -> tuple[AgentMiddleware, ...]:
        """Create fresh middleware for a peer or isolated child's message state.

        Use LangChain's public middleware directly: it replaces retained history
        with a summary and recent messages. No DeepAgents default compactor is
        installed in this sample, so there is exactly one compaction policy.
        The summary adapter is explicit to separate its scripted answer cursor
        and accounting from agent decisions. Live callers may use the same model.
        """
        return (
            SummarizationMiddleware(
                model=summary_model,
                trigger=("tokens", self.trigger_tokens),
                keep=("tokens", self.keep_tokens),
                # Fix the demonstration to one documented local estimator;
                # simulated provider receipts must not rescale this threshold.
                token_counter=lambda messages: count_tokens_approximately(messages),
                trim_tokens_to_summarize=None,
            ),
            InputBudgetMiddleware(self.max_input_tokens),
        )
