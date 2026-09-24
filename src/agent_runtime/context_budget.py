"""Apply workflow-selected history compaction and an estimated input ceiling.

One immutable budget can configure multiple peers, but their workflow must also
pass shared messages. A child gets a separate budget and separate middleware.
Triggers use validated input plus estimated retained output and new messages.
Fallback counts and envelope adjustments are local estimates, not tokenizer guarantees. The ceiling
covers agent requests, including instructions/tools; summary-model calls have
their own provider limits. Output allowance must be reserved by the caller.
Successful compactions emit count-only callback evidence for report projections.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from math import ceil

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
    dispatch_custom_event,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool


def estimated_retained_output_tokens(output_tokens, reasoning_tokens, content):
    """Estimate how much of one response can enter the next request.

    Provider output includes generated reasoning, but that reasoning only
    remains in this application's message history when the adapter returns a
    reasoning item. An encrypted item is carried forward without exposing its
    private text. Its exact next-input size is unavailable, so use the reported
    reasoning tokens as an estimate. If no item is retained, use reported
    non-reasoning output (which can include formatting overhead). The next
    provider input receipt supersedes this estimate.
    """
    has_reasoning_item = isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in {"reasoning", "thinking"}
        for block in content
    )
    return output_tokens if has_reasoning_item else max(0, output_tokens - reasoning_tokens)


def _local_history_estimate(messages):
    """Estimate carried history without tokenizing encrypted reasoning bytes.

    Ciphertext length does not measure the model's reasoning tokens. Count the
    ordinary message shape and add the reported reasoning count for retained
    reasoning items when present. Missing usage leaves that portion unknown;
    the separate input guard still checks its local estimate before the call.
    """
    visible = []
    retained_reasoning = 0
    for message in messages:
        if not isinstance(message, AIMessage) or not isinstance(message.content, list):
            visible.append(message)
            continue
        blocks = []
        encrypted_reasoning = False
        for block in message.content:
            if (isinstance(block, dict) and block.get("type") in {"reasoning", "thinking"}
                    and block.get("encrypted_content")):
                encrypted_reasoning = True
                block = {key: value for key, value in block.items()
                         if key != "encrypted_content"}
            blocks.append(block)
        if encrypted_reasoning:
            details = (message.usage_metadata or {}).get("output_token_details") or {}
            if isinstance(details.get("reasoning"), int):
                retained_reasoning += details["reasoning"]
        visible.append(message.model_copy(update={"content": blocks}))
    return count_tokens_approximately(visible) + retained_reasoning


def _history_fingerprint(messages):
    """Identify the retained message prefix that a usage receipt actually measured.

    Graph reducers assign IDs and tracing adds metadata after a model call, so
    neither belongs in this comparison. Content and tool exchanges do: removing
    or rewriting one of them invalidates the old input total. Store only a digest
    in response metadata, never another copy of the user's conversation.
    """
    records = [
        {"type": message.type, "content": message.content,
         "name": message.name, "tool_calls": getattr(message, "tool_calls", None),
         "tool_call_id": getattr(message, "tool_call_id", None)}
        for message in messages
    ]
    return sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def context_estimate(messages):
    """Use the latest valid input and retained output, plus new messages.

    Input already includes cached tokens, instructions and tool definitions;
    adding cache counts or that same envelope again would double-count them.
    A retained reasoning item contributes an estimated reasoning-token amount;
    reasoning without such an item is excluded from next-request history. This
    is not an exact count of retained input. A peer may also change
    its instructions/tools. The separate final-input guard checks that request.

    Compaction changes the prefix. A receipt from a surviving assistant message
    then describes deleted content and must be discarded. Until fresh usage is
    available, use the local message estimate and the last known envelope.
    The returned basis makes this fallback visible in report evidence.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, AIMessage):
            continue
        receipt = message.response_metadata.get("context_budget_receipt")
        usage = message.usage_metadata
        if not isinstance(receipt, dict):
            continue
        if receipt.get("prefix") != _history_fingerprint(messages[:index]):
            # Once the latest measured prefix differs, older receipts cannot
            # establish the size of this rewritten conversation either.
            return (_local_history_estimate(messages) + receipt["envelope_tokens"],
                    "local estimate after history replacement")
        if usage is not None and all(type(usage.get(key)) is int and usage[key] >= 0
                                     for key in ("input_tokens", "output_tokens")):
            added = _local_history_estimate(messages[index + 1:])
            reasoning = (usage.get("output_token_details") or {}).get("reasoning", 0)
            reasoning = reasoning if isinstance(reasoning, int) else 0
            retained_output = estimated_retained_output_tokens(
                usage["output_tokens"], reasoning, message.content
            )
            return (usage["input_tokens"] + retained_output + added,
                    "reported input + estimated retained output"
                    + (" + estimated new messages" if added else ""))
    return _local_history_estimate(messages), "local estimate; no valid usage yet"


def _envelope_tokens(request):
    """Estimate only the role-specific instructions and advertised tool schemas.

    These are injected for this request rather than saved in shared history.
    A switch from planner to responder can change this cost even when both use
    exactly the same messages. Keep serialization consistent for both receipts
    and guard adjustments so an unchanged envelope cancels out.
    """
    tokens = count_tokens_approximately([request.system_message]) if request.system_message else 0
    if request.tools:
        schemas = [convert_to_openai_tool(tool) for tool in request.tools]
        tokens += ceil(len(json.dumps(schemas, ensure_ascii=False)) / 4)
    return tokens


class ReportingSummarizationMiddleware(SummarizationMiddleware):
    """Trigger native history replacement from the usage-based context estimate.

    Emit counts only after a successful replacement. No-op checks and failed
    summary calls must not be presented as completed compactions. Counts use the
    trigger's estimator. Keep/cutoff selection still uses local message sizes:
    a full-request receipt cannot describe the size of a candidate suffix.
    """

    def __init__(self, *, max_input_tokens, **kwargs):
        """Keep the separate input ceiling alongside the native summary policy."""
        super().__init__(**kwargs)
        self.max_input_tokens = max_input_tokens

    @property
    def name(self):
        """Preserve the native middleware node name in existing trace views."""
        return "SummarizationMiddleware"

    def _should_summarize(self, messages, total_tokens):
        """Use one explicit token threshold, without native stale-receipt fallback.

        ContextBudget installs only a token trigger. Native middleware otherwise
        ORs its local count with the last reported total even after replacement;
        validating the receipt prevents repeated compaction of deleted history.
        """
        return context_estimate(messages)[0] >= self.trigger[1]

    def _compaction_evidence(self, before, before_count, update):
        """Describe the replacement, excluding its reducer deletion sentinel."""
        retained = update["messages"][1:]
        after = context_estimate(retained)
        return {
            "compaction_event": "completed",
            "compaction_before_tokens": before[0],
            "compaction_after_tokens": after[0],
            "compaction_before_basis": before[1],
            "compaction_after_basis": after[1],
            "compaction_before_messages": before_count,
            "compaction_after_messages": len(retained),
            "compaction_trigger_tokens": self.trigger[1],
            "compaction_keep_tokens": self.keep[1],
            "compaction_max_input_tokens": self.max_input_tokens,
        }

    def before_model(self, state, runtime):
        """Record successful synchronous compaction on the active callback scope."""
        before = context_estimate(state["messages"])
        count = len(state["messages"])
        update = super().before_model(state, runtime)
        if update is not None:
            dispatch_custom_event(
                "context_compaction", self._compaction_evidence(before, count, update)
            )
        return update

    async def abefore_model(self, state, runtime):
        """Capture identical evidence for asynchronous browser/AG-UI execution."""
        before = context_estimate(state["messages"])
        count = len(state["messages"])
        update = await super().abefore_model(state, runtime)
        if update is not None:
            await adispatch_custom_event(
                "context_compaction", self._compaction_evidence(before, count, update)
            )
        return update


class ContextBudgetExceeded(ValueError):
    """Stop before an oversized agent request instead of silently losing content."""


class InputBudgetMiddleware(AgentMiddleware):
    """Check the complete agent request after history summarization has run."""

    def __init__(self, limit: int):
        self.limit = limit

    def check(self, request):
        """Adjust the usage baseline for this request's instructions and tools.

        The latest message or an indivisible tool exchange may exceed the budget
        even after summarization. Reject it visibly; do not truncate user intent
        or send a request that violates the declared estimate. Counting the full
        message text again would lose the measured baseline and can badly
        overcount encoded reasoning blocks returned by a provider.
        """
        messages = list(request.messages)
        tokens, _ = context_estimate(messages)
        previous_envelope = 0
        for message in reversed(messages):
            receipt = message.response_metadata.get("context_budget_receipt")
            if isinstance(receipt, dict):
                previous_envelope = receipt["envelope_tokens"]
                break
        # The baseline (and a replacement fallback) already includes the last
        # measured request's envelope. Replace that estimated contribution; do
        # not add a second entire envelope when peers hand off shared history.
        tokens = max(0, tokens - previous_envelope) + _envelope_tokens(request)
        if tokens > self.limit:
            raise ContextBudgetExceeded(
                f"Estimated agent input {tokens} tokens exceeds budget {self.limit}; "
                "shorten the input or raise this workflow/subagent budget."
            )

    def wrap_model_call(self, request, handler):
        """Guard synchronous calls after before-model compaction updates state."""
        self.check(request)
        return self._record_receipt(request, handler(request))

    async def awrap_model_call(self, request, handler):
        """Apply the identical guard to browser/AG-UI asynchronous execution."""
        self.check(request)
        return self._record_receipt(request, await handler(request))

    def _record_receipt(self, request, response):
        """Bind provider usage to its history so shared peers can safely reuse it.

        Response metadata stays in application state; it is not a system prompt
        or a message appended to the shared history. Sync and async paths stamp
        the same evidence. Missing usage remains missing; we never invent totals.
        """
        envelope = _envelope_tokens(request)
        for message in response.result:
            if isinstance(message, AIMessage) and message.usage_metadata is not None:
                message.response_metadata = {
                    **message.response_metadata,
                    "context_budget_receipt": {
                        "prefix": _history_fingerprint(request.messages),
                        "envelope_tokens": envelope,
                    },
                }
        return response


@dataclass(frozen=True)
class ContextBudget:
    """Configure one conversation's policy, independently of model capacity.

    trigger_tokens uses input plus retained-output and new-message estimates;
    keep_tokens targets recent unsummarized history. Whole messages/tool pairs can exceed it.
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

        Delegate replacement to LangChain's public middleware and emit count-only
        evidence from its lifecycle hooks. No DeepAgents default compactor is
        installed in this sample, so there is exactly one compaction policy.
        The summary adapter is explicit to separate its scripted answer cursor
        and accounting from agent decisions. Live callers may use the same model.
        """
        return (
            ReportingSummarizationMiddleware(
                max_input_tokens=self.max_input_tokens,
                model=summary_model,
                trigger=("tokens", self.trigger_tokens),
                keep=("tokens", self.keep_tokens),
                # Local counts select which whole messages to keep. The trigger
                # above separately uses validated usage for the complete context.
                token_counter=lambda messages: count_tokens_approximately(messages),
                trim_tokens_to_summarize=None,
            ),
            InputBudgetMiddleware(self.max_input_tokens),
        )
