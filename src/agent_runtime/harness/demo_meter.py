"""Simulate token counts and cache reuse for offline agent examples.

No model provider is called. tiktoken's o200k_base encoding counts a stable JSON
representation of tool definitions and messages, including message framing.
These are tokenizer-based estimates of our representation, not exact provider
request counts. The vocabulary may download on first use and is cached by
tiktoken; prepare that cache before running samples without network access.

ContextSimulation keeps a separate history for each simulated model. It assumes
completed visible conversation is cached immediately and never expires, and
rejects requests that drop earlier context. Its ledger explains fresh input,
cache reads, and context growth; cache movement is not a billable cache write.
Live requests use provider-reported usage instead of these simulated counts.
Message serialization helpers also let reports compare the same representation.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

import tiktoken

# One encoding keeps all offline lessons and report detail counts comparable.
# Scripted models have no provider tokenizer identity to select dynamically.
TOKEN_ENCODING = "o200k_base"
TOKEN_ESTIMATE_BASIS = f"Estimated with tiktoken {TOKEN_ENCODING} over canonical JSON"


def message_record(message):
    """Give capture and simulated accounting the same message representation.

    ``message`` is a LangChain message-like object. Used by both capture and
    simulation so displayed messages and measured
    simulated context share a representation. Tool-call JSON counts as model
    output; its matching tool result is a separate message in the next request.
    """
    # AIMessage.tool_calls contains proposed tool names and arguments. A later
    # ToolMessage carries executed output in content and links it through
    # tool_call_id; keeping both avoids treating the proposal as a tool result.
    return {
        "role": message.type,
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def units(value):
    """Return token IDs for canonical JSON through the shared sample estimator.

    Stable key ordering makes equivalent dictionaries count the same way.
    Encode each message separately so appending another message cannot change
    its predecessor's token boundaries; the cache ledger needs that stable prefix.
    This adds our JSON framing, not a provider-specific chat envelope. Ordinary
    encoding treats special-token-looking text as literal user data instead of
    rejecting it or interpreting it as a control token.

    tiktoken caches the encoding in memory and its vocabulary on disk. First use
    can download that public vocabulary; encoding sends no conversation data.
    Loading errors propagate rather than silently switching counting methods.
    """
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return tiktoken.get_encoding(TOKEN_ENCODING).encode_ordinary(text)


def message_units(record):
    """Use the central tokenizer for one serialized message, including its framing."""
    return units(record)


class ContextSimulation:
    """Model growing context and ideal cache reuse for one simulated agent.

    Each model instance owns its own simulation: a specialist must not inherit
    its parent's cache. record(definitions, messages, response) returns a usage
    ledger entry and role totals, then retains the completed visible conversation
    for the next call. This deliberately assumes immediate caching without expiry;
    real provider usage must come from its SDK instead of this teaching model.
    """

    def __init__(self):
        """Start an empty context/cache and an audit ledger for one conversation."""
        # Lists contain the canonical counting units, not messages. Keeping the
        # actual prefix lets record() reject a changed history of the same length.
        self.context = []
        self.cached = []
        self.ledger = []

    def record(self, definitions, messages, response):
        """Explain an offline call's context growth and idealized cache reuse.

        ``definitions`` are bound tool schemas; ``messages`` and ``response`` use
        ``message_record``. The returned tuple is ``(usage_entry, role_counts)``.
        All earlier visible context must be retained, or ValueError exposes a
        simulation that no longer matches its append-only teaching assumption.
        Cache-write ledger entries describe context movement, not provider-billed
        cache creation: charging them requires an explicit provider usage field.
        """
        # Tool definitions precede messages and consume input even when the user
        # prompt is short; leaving them out makes the first request misleading.
        request_units = units(definitions)
        roles = {"definitions": len(request_units)}
        for message in messages:
            message_token_units = message_units(message)
            role = message["role"]
            roles[role] = roles.get(role, 0) + len(message_token_units)
            request_units.extend(message_token_units)
        # Every token of the completed prior request+response must remain the
        # next request prefix. A mismatch means history was dropped or rewritten
        # (or definitions changed), violating this append-only simulation. Reject
        # it instead of claiming cache hits for content no longer being sent.
        if request_units[: len(self.context)] != self.context:
            raise ValueError(
                "Simulation requires previous context and response to be retained"
            )
        response_units = message_units(response)
        usage_entry = {
            "request_tokens": len(request_units),
            "cache_read_tokens": len(self.cached),
            "fresh_input_tokens": len(request_units) - len(self.cached),
            "request_cache_write_tokens": len(request_units) - len(self.cached),
            "response_cache_write_tokens": len(response_units),
            "added_message_tokens": len(request_units) - len(self.context),
            "response_tokens": len(response_units),
            "context_after_response_tokens": len(request_units) + len(response_units),
        }
        # Only a successful prefix check advances the simulation. The completed
        # visible exchange becomes both next-call context and immediately cached
        # content; no provider cache service is contacted or confirmed.
        self.context = request_units + response_units
        self.cached = self.context.copy()
        self.ledger.append(usage_entry)
        return usage_entry, roles
