"""Observe message-history changes separately from provider billing/cache usage."""

from collections import Counter


def context_change(step, previous):
    current = step.request
    old = previous.request if previous else []
    retained = 0
    for before, after in zip(old, current):
        if before != after:
            break
        retained += 1
    tail = current[retained:]
    roles = Counter(m.get("role", "unknown") for m in tail)
    names = {
        "human": "user messages",
        "user": "user messages",
        "ai": "assistant messages",
        "assistant": "assistant messages",
        "tool": "tool results",
        "system": "system messages",
    }
    additions = [f"{count} {names.get(role, role)}" for role, count in roles.items()]
    calls = sum(len(m.get("tool_calls", [])) for m in tail)
    if calls:
        additions.append(f"{calls} tool calls in assistant messages")
    return {
        "input_tokens": step.usage.input_tokens if step.usage else None,
        "previous_tokens": previous.usage.input_tokens
        if previous and previous.usage
        else None,
        "delta": step.usage.input_tokens - previous.usage.input_tokens
        if previous and previous.usage and step.usage
        else None,
        "message_count": len(current),
        "retained": retained,
        "changed": bool(previous and retained < len(old)),
        "additions": additions,
        "available": bool(current),
        "composition": [
            (key.removeprefix("simulated_").removesuffix("_tokens"), value)
            for key, value in step.context.items()
            if key.startswith("simulated_")
        ],
    }
