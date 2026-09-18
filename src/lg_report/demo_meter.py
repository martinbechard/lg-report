"""Deterministic simulation units; never used to price real provider requests."""

import json
import re


def message_record(message):
    return {
        "role": message.type,
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def units(value):
    """Count words and punctuation in canonical JSON, including message framing."""
    return re.findall(
        r"\w+|[^\w\s]", json.dumps(value, sort_keys=True, ensure_ascii=False)
    )


def message_units(record):
    return units(record)


class ContextSimulation:
    """Append-only conversation with a cache of the completed conversation."""

    def __init__(self):
        self.context = []
        self.cached = []
        self.ledger = []

    def record(self, definitions, messages, response):
        prompt = units(definitions)
        roles = {"definitions": len(prompt)}
        for message in messages:
            tokens = message_units(message)
            role = message["role"]
            roles[role] = roles.get(role, 0) + len(tokens)
            prompt.extend(tokens)
        if prompt[: len(self.context)] != self.context:
            raise ValueError(
                "Simulation requires previous context and response to be retained"
            )
        output = message_units(response)
        entry = {
            "request_tokens": len(prompt),
            "cache_read_tokens": len(self.cached),
            "fresh_input_tokens": len(prompt) - len(self.cached),
            "request_cache_write_tokens": len(prompt) - len(self.cached),
            "response_cache_write_tokens": len(output),
            "added_message_tokens": len(prompt) - len(self.context),
            "response_tokens": len(output),
            "context_after_response_tokens": len(prompt) + len(output),
        }
        self.context = prompt + output
        self.cached = self.context.copy()
        self.ledger.append(entry)
        return entry, roles
