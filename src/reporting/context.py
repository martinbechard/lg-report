"""Describe how visible message history changes between successive model requests.

The report measures request occupancy and compares retained prefixes and newly
added message roles to explain context growth. Callers must compare requests on
the same agent path, otherwise a subagent's separate conversation would look
like deleted or replaced history.

Matching visible messages does not prove a provider cache hit. This module keeps
message comparisons separate from the provider's token usage and billing data.
Explicit compaction events carry their own local estimates and policy limits.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from collections import Counter


def compaction_description(step):
    """Explain explicit replacement evidence without inferring missing counts.

    This text is shared by diagram tooltips and HTML/Excel execution tables.
    Local history estimates are never added to provider usage or billed tokens.
    Older traces have no event and receive no fabricated compaction annotation.
    """
    context = step.context
    if context.get("compaction_event") != "completed":
        return None

    def count(field):
        """Retain unavailable values rather than displaying them as zero."""
        value = context.get(f"compaction_{field}")
        return f"{value:,}" if isinstance(value, int) else "unreported"

    # New recordings disclose each counter's basis instead of suggesting that
    # a provider receipt and a post-replacement local estimate are identical.
    # Keep legacy recordings truthful: their numbers remain history-only counts.
    if "compaction_before_basis" in context:
        return (
            f"Compaction completed | Trigger context: {count('before_tokens')} tokens "
            f"({context['compaction_before_basis']}) | "
            f"After replacement: {count('after_tokens')} tokens "
            f"({context['compaction_after_basis']}) | "
            f"Compact at: {count('trigger_tokens')} context tokens | "
            f"Input error limit: {count('max_input_tokens')} estimated tokens"
        )

    # Completion means history was replaced, not that the replacement shrank.
    # Small histories can grow when summary prose and its wrapper exceed the
    # removed content, while recent tool exchanges must still be retained.
    before = context.get("compaction_before_tokens")
    after = context.get("compaction_after_tokens")
    effect = ""
    if isinstance(before, int) and isinstance(after, int):
        delta = after - before
        effect = (
            f"Estimated history grew by {delta:,} tokens; no size reduction"
            if delta > 0 else
            f"Estimated history reduced by {-delta:,} tokens"
            if delta < 0 else "Estimated history size unchanged"
        ) + " | "
    return (
        "Compaction completed | Estimated history tokens: "
        f"{count('before_tokens')} → {count('after_tokens')} | "
        f"{effect}"
        f"Messages: {count('before_messages')} → {count('after_messages')} | "
        f"Trigger threshold: {count('trigger_tokens')} tokens | "
        f"Recent-history retention target: {count('keep_tokens')} tokens | "
        f"Maximum agent input: {count('max_input_tokens')} estimated tokens "
        "(separate guard, includes system instructions and tool definitions) | "
        "History estimates exclude system/tool envelopes; retained history includes the summary. "
        "Trigger may also use matching-provider reported usage."
    )


def context_utilization(step, prices, *, include_output=False):
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
    # The request view measures input alone; the history plot shows occupancy
    # after this response. Output is already inclusive of reasoning. Neither
    # mode adds cached tokens again: they are part of usage.input_tokens.
    tokens = step.usage.input_tokens + (step.usage.output_tokens if include_output else 0)
    return {
        **capacity,
        "percent": tokens / capacity["capacity"] * 100,
        "tokens": tokens,
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
    # An explicit check overrides the legacy table, including failed checks.
    # Saved Prices snapshots carry this evidence, so rendering needs no network.
    checked = prices.calibrations.get(key)
    if checked is not None:
        if checked.capacity is None or checked.error:
            return None
        return {
            "capacity": checked.capacity,
            "model": key,
            "source": checked.source,
            "illustrative": illustrative,
        }
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
