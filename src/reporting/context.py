"""Describe how visible message history changes between successive model requests.

The report measures request occupancy and compares retained prefixes and newly
added message roles to explain context growth. Callers must compare requests on
the same agent path, otherwise a subagent's separate conversation would look
like deleted or replaced history.

Matching visible messages does not prove a provider cache hit. This module keeps
message comparisons separate from the provider's token usage and billing data.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from collections import Counter


def context_utilization(step, prices):
    """Measure request occupancy against a verified model context capacity.

    Input usage already includes cached tokens. Output is excluded because this
    measures the context at request time, not its size after generation. Exact
    identities and explicit aliases avoid guessing a capacity for unknown models.
    Demo tariffs may name a real model basis; that yields an illustrative ratio,
    never a claim that the simulator has a provider-enforced context limit.
    """
    capacity = context_capacity(step.provider, step.model, prices)
    if capacity is None or step.usage is None:
        return None
    return {
        **capacity,
        "percent": step.usage.input_tokens / capacity["capacity"] * 100,
        "tokens": step.usage.input_tokens,
    }


def context_capacity(provider, model, prices):
    """Resolve the shared capacity independently of measured or estimated usage."""
    # Official capacities verified 2026-09-20. Keep the source with each point
    # so an exported report exposes the denominator even without network access.
    openai = "https://developers.openai.com/api/docs/models/"
    claude = "https://platform.claude.com/docs/en/models/"
    capacities = {
        **{
            f"openai:{model}": (1_050_000, f"{openai}{model}")
            for model in ("gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol")
        },
        **{
            f"anthropic:{model}": (1_000_000, f"{claude}overview")
            for model in ("claude-sonnet-5", "claude-fable-5-1")
        },
        "anthropic:claude-opus-4-8": (
            1_000_000,
            "https://platform.claude.com/docs/de/models/opus-4-8/overview",
        ),
    }
    key = f"{provider}:{model}"
    key = prices.aliases.get(key, key)
    illustrative = provider == "demo"
    if illustrative:
        rate = prices.models.get(key)
        key = rate.based_on if rate else None
    capacity = capacities.get(key)
    if capacity is None:
        return None
    tokens, source = capacity
    return {
        "capacity": tokens,
        "model": key,
        "source": source,
        "illustrative": illustrative,
    }


def context_change(step, previous):
    """Help report readers understand why the next model request grew or changed.

    Return display-ready message additions and measured token differences for
    ``step``, the current normalized model Step. This is a comparison of visible
    inputs, not an explanation of provider internals or proof of cache hits.

    previous must be the prior model step on the same conversation path (or
    None), not simply the preceding model call from another subagent. Equality
    identifies retained message prefixes; it cannot establish cache billing or
    exact token contributions. Missing usage produces None token deltas.
    """
    current_messages = step.request
    # The first call on this path has no history baseline; another agent
    # must not supply one merely because its call occurred earlier in time.
    previous_messages = previous.request if previous else []
    retained_prefix_count = 0
    for before, after in zip(previous_messages, current_messages):
        # Cacheable continuity ends at the first changed message. Later equal
        # messages cannot repair the broken prefix, so stop counting there.
        if before != after:
            break
        retained_prefix_count += 1
    added_messages = current_messages[retained_prefix_count:]
    roles = Counter(m.get("role", "unknown") for m in added_messages)
    role_labels = {
        "human": "user messages",
        "user": "user messages",
        "ai": "assistant messages",
        "assistant": "assistant messages",
        "tool": "tool results",
        "system": "system messages",
    }
    additions = [
        f"{count} {role_labels.get(role, role)}" for role, count in roles.items()
    ]
    tool_call_count = sum(len(m.get("tool_calls", [])) for m in added_messages)
    # Tool requests are embedded in assistant messages, so name their presence
    # separately when any exist; avoid a distracting zero-call annotation.
    if tool_call_count:
        additions.append(f"{tool_call_count} tool calls in assistant messages")
    # Current tokens require current usage. Previous tokens additionally need a
    # prior call and its usage; a delta needs all three. None preserves missing
    # evidence, whereas zero would misleadingly assert a measured count/change.
    # A changed prefix requires a prior call and fewer retained messages than
    # that prior request; simply appending messages is not replacement.
    return {
        "input_tokens": step.usage.input_tokens if step.usage else None,
        "previous_tokens": previous.usage.input_tokens
        if previous and previous.usage
        else None,
        "delta": step.usage.input_tokens - previous.usage.input_tokens
        if previous and previous.usage and step.usage
        else None,
        "message_count": len(current_messages),
        "retained": retained_prefix_count,
        "changed": bool(previous and retained_prefix_count < len(previous_messages)),
        "additions": additions,
        "available": bool(current_messages),
        # Only simulation-owned counters carry the simulated_ prefix. Other
        # metadata is not a token contribution and must not enter composition.
        "composition": [
            (key.removeprefix("simulated_").removesuffix("_tokens"), value)
            for key, value in step.context.items()
            if key.startswith("simulated_")
        ],
    }
