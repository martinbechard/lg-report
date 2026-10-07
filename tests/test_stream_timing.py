"""Verify client streaming timing without paid calls or storing private chunks.

Empty lifecycle events must not look like generated tokens. Controlled clocks
make first-output attribution and per-call isolation deterministic.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, LLMResult

from agent_runtime.harness.model_providers import get_provider
from agent_runtime.harness.trace_capture import TraceCapture
from reporting.native_timing import native_timing
from reporting.normalize import normalize
from reporting.schema import Step


def test_openai_streams_for_invoke():
    """Ordinary invoke must enable streaming while retaining native Responses."""
    model = get_provider("openai").create_model("fixture", {"OPENAI_API_KEY": "placeholder"})
    assert model.streaming and model.use_responses_api


@pytest.mark.parametrize("chunk", [
    AIMessageChunk(content="PRIVATE"),
    AIMessageChunk(content="", tool_call_chunks=[{"args": '{"x":', "index": 0}]),
    AIMessageChunk(content=[{"type": "reasoning", "summary": [{"text": "PRIVATE"}]}]),
])
def test_first_output_survives_metadata_only_capture(monkeypatch, tmp_path, chunk):
    """Headers are ignored, first meaningful output wins, and calls stay isolated."""
    clock = iter([0, 100, 200, 300, 400])
    monkeypatch.setattr("agent_runtime.harness.trace_capture.perf_counter_ns", lambda: next(clock))
    capture = TraceCapture(tmp_path / "spans.jsonl", "openai", "fixture")
    call = uuid4()
    try:
        capture.on_chat_model_start({}, [[HumanMessage("PRIVATE")]], run_id=call)
        capture.on_llm_new_token("", run_id=call, chunk=ChatGenerationChunk(message=AIMessageChunk(
            content=[{"type": "reasoning", "id": "item", "summary": []}])))
        capture.on_llm_new_token(chunk.text, run_id=call, chunk=ChatGenerationChunk(message=chunk))
        capture.on_llm_new_token("Later", run_id=call)
        capture.on_llm_end(LLMResult(generations=[]), run_id=call)
        # A subsequent non-streaming call has no first-output observation.
        other = uuid4()
        capture.on_chat_model_start({}, [[HumanMessage("PRIVATE")]], run_id=other)
        capture.on_llm_end(LLMResult(generations=[]), run_id=other)
    finally:
        capture.close()
    run = normalize(tmp_path / "spans.jsonl", title="Client timing")
    assert run.steps[0].context["client_ttft_ns"] == 200
    assert native_timing(run.steps[0])["ttft_ms"] == .0002
    assert native_timing(run.steps[1])["ttft_ms"] is None
    assert not capture.model_started
    assert "PRIVATE" not in run.model_dump_json()


def test_api_unknowns_and_failed_calls():
    """Elapsed time is measured; internal breakdowns and invalid TTFT stay unknown."""
    step = Step(id="api", name="api", kind="model", provider="openai", status="ok",
                start_ns=0, end_ns=1_000_000_000, context={"client_ttft_ns": 250_000_000})
    values = native_timing(step)
    assert values["turn_ms"] == 1000 and values["ttft_ms"] == 250
    assert set(values) == {"turn_ms", "ttft_ms"}
    for invalid in [-1, 2_000_000_000, True, "250"]:
        assert native_timing(step.model_copy(update={"context": {"client_ttft_ns": invalid}}))["ttft_ms"] is None
    assert all(value is None for value in native_timing(step.model_copy(update={"status": "error"})).values())
