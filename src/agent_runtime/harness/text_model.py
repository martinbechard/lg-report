"""Share the text-only LangChain contract used by local login transports.

Codex and Copilot keep their own process/session lifecycle and receipt parsing.
This module owns history validation, trace identity, tool-binding policy, and
inclusive token accounting. Native API adapters already implement LangChain's
richer contract and do not inherit these text-only restrictions.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from typing import ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def text_result(content: str, metadata: dict, *, usage: dict | None = None) -> ChatResult:
    """Package a local receipt without inventing missing totals or cache counts.

    Transports translate their field names into inclusive input/output totals,
    cache_read_input_tokens, cache_creation_input_tokens, and optional
    reasoning_output_tokens. Raw receipts stay in metadata for auditability.
    Cache/reasoning counts are subsets, never additional tokens.
    """
    receipt = usage if usage is not None else metadata.get("usage", {})
    label = metadata["provider"].title()
    metered = None
    if receipt.get("input_tokens") is not None and receipt.get("output_tokens") is not None:
        for key, value in receipt.items():
            if key.endswith("tokens") and (type(value) is not int or value < 0):
                raise ValueError(f"{label} returned invalid token usage")
        metered = {"input_tokens": receipt["input_tokens"],
                   "output_tokens": receipt["output_tokens"],
                   "total_tokens": receipt["input_tokens"] + receipt["output_tokens"]}
        details = {target: receipt[source] for source, target in (
            ("cache_read_input_tokens", "cache_read"),
            ("cache_creation_input_tokens", "cache_creation"),
        ) if source in receipt}
        if sum(details.values()) > receipt["input_tokens"]:
            raise ValueError(f"{label} cache usage exceeds input tokens")
        if details:
            metered["input_token_details"] = details
        if "reasoning_output_tokens" in receipt:
            if receipt["reasoning_output_tokens"] > receipt["output_tokens"]:
                raise ValueError(f"{label} reasoning usage exceeds output tokens")
            metered["output_token_details"] = {"reasoning": receipt["reasoning_output_tokens"]}
    return ChatResult(generations=[ChatGeneration(message=AIMessage(
        content=content, response_metadata=metadata, usage_metadata=metered,
    ))])


class TextOnlyChatModel(BaseChatModel):
    """Common local-model behavior; subclasses implement sync/async transport.

    LangGraph owns history. Tool schemas are suppressed, while forced tools and
    unrepresentable inputs fail before a request. No retries or model fallback
    are added here. These constraints do not apply to native API models.
    """

    provider: ClassVar[str]
    model_name: str
    reasoning_effort: str | None = None
    request_timeout: float = Field(default=120, gt=0)

    @property
    def _llm_type(self) -> str:
        """Identify the transport separately from its underlying model vendor."""
        return self.provider

    def _get_ls_params(self, stop=None, **kwargs):
        """Use the same fixed model identity for reporting and LangSmith."""
        return {"ls_provider": self.provider, "ls_model_name": self.model_name,
                "ls_model_type": "chat"}

    @property
    def _identifying_params(self):
        """Keep the selected effort visible in invocation accounting."""
        return {"model_name": self.model_name, "reasoning_effort": self.reasoning_effort}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Accept ordinary graph construction, but reject required tool work."""
        if tool_choice not in (None, "auto", "none"):
            raise ValueError(f"{self.provider.title()} request/response mode disables graph tools")
        return self

    def prepare_history(self, messages, stop, kwargs) -> tuple[str, str]:
        """Validate once and encode explicit roles for either local transport."""
        label = self.provider.title()
        if stop or kwargs:
            raise ValueError(f"{label} adapter does not support stop or invocation overrides")
        system, history = [], []
        for message in messages:
            if not isinstance(message.content, str):
                raise TypeError(f"{label} request/response supports text messages only")
            if message.type == "system":
                system.append(message.content)
            elif message.type in {"human", "ai"} and not getattr(message, "tool_calls", None):
                history.append({"role": "user" if message.type == "human" else "assistant",
                                "content": message.content})
            else:
                raise ValueError(f"{label} request/response does not accept tool messages")
        instructions = "\n\n".join(system) or "Answer the user's request."
        instructions += "\nThe prompt is JSON conversation history. Answer its final user turn directly."
        return instructions, json.dumps(history, ensure_ascii=False)
