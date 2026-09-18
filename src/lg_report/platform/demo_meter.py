"""Simulate token counts and cache reuse for offline agent examples.

No provider is called. The simulator counts words and punctuation in a stable
JSON representation of tool definitions and messages, including message framing.
These are repeatable teaching estimates, not an OpenAI or Anthropic tokenizer.

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
import re


def message_record(message):
    """Retain message content and tool linkage without SDK implementation fields.

    Used by both capture and simulation so displayed messages and measured
    simulated context share a representation. Tool-call JSON counts as model
    output; its matching tool result is a separate message in the next request.
    """
    return {
        "role": message.type,
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def units(value):
    """Return deterministic words/punctuation, including JSON message framing.

    These are teaching units, not a provider tokenizer. Stable key ordering lets
    successive calls compare context prefixes despite dictionary insertion order.
    """
    return re.findall(
        r"\w+|[^\w\s]", json.dumps(value, sort_keys=True, ensure_ascii=False)
    )


def message_units(record):
    """Apply the same counting rule to incoming messages and outgoing responses."""
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
        """Account for a request and append its response to this agent's cache.

        definitions are bound tool schemas; messages/response use message_record.
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
        self.context = request_units + response_units
        self.cached = self.context.copy()
        self.ledger.append(usage_entry)
        return usage_entry, roles
