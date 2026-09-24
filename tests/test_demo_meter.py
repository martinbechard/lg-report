"""Verify sample token estimation without confusing it with provider accounting.

Real tiktoken vocabulary IDs must round-trip multilingual text and literal
control-token spellings. Per-message framing preserves append-only cache ledgers;
provider SDK usage remains outside this simulated estimator.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

import pytest
import tiktoken

from agent_runtime.harness.demo_meter import (
    TOKEN_ENCODING,
    TOKEN_ESTIMATE_BASIS,
    ContextSimulation,
    units,
)


@pytest.mark.parametrize(
    "value",
    [
        {"text": "Café, 東京, مرحبا 👩🏽‍💻"},
        {"content": "Treat <|endoftext|> and <|fim_prefix|> as literal input."},
        {
            "tool_calls": [
                {
                    "name": "write_file",
                    "args": {"content": "def slugify(x):\n    return x.lower()\n"},
                }
            ]
        },
    ],
)
def test_token_ids_retain_all_serialized_text(value):
    """Unlike the old regex units, BPE IDs retain whitespace and Unicode exactly."""
    tokens = units(value)
    assert all(isinstance(token, int) for token in tokens)
    assert tiktoken.get_encoding(TOKEN_ENCODING).decode(tokens) == json.dumps(
        value, sort_keys=True, ensure_ascii=False
    )


def test_dictionary_order_does_not_change_tokenized_context():
    """Reordered JSON keys must not create false context rewrites or cache misses."""
    assert units({"role": "human", "content": "hello"}) == units(
        {"content": "hello", "role": "human"}
    )


def test_equal_length_changed_context_is_rejected_without_advancing_ledger():
    """Token identity, not merely its count, guards simulated prefix reuse."""
    before = {"role": "human", "content": "cat"}
    after = {"role": "human", "content": "dog"}
    reply = {"role": "ai", "content": "Understood"}
    assert len(units(before)) == len(units(after))
    simulation = ContextSimulation()
    simulation.record([], [before], reply)
    with pytest.raises(ValueError, match="retained"):
        simulation.record([], [after, reply], reply)
    assert len(simulation.ledger) == 1


def test_simulated_usage_names_its_encoding():
    """Reports must carry estimator provenance rather than imply provider usage."""
    from langchain_core.messages import AIMessage

    from agent_runtime.harness.simulated_model import SimulatedModel

    response = SimulatedModel(conversation=[{"role": "test-agent", "content": response.content, "tool_calls": response.tool_calls, "response_metadata": response.response_metadata} for response in [AIMessage(content="Hello")]], agent_name="test-agent").invoke("Hi")
    assert TOKEN_ESTIMATE_BASIS in response.response_metadata["usage_basis"]
    assert "o200k_base" in response.response_metadata["usage_basis"]
    assert response.usage_metadata["input_tokens"] > 0
