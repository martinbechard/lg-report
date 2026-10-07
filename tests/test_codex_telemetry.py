"""Verify native timing capture, attribution, privacy, and unknown measurements.

Synthetic OTLP packets exercise the actual loopback receiver without invoking
models. Delta/cumulative exports must not multiply timing or token costs.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from urllib.request import Request, urlopen

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent_runtime.harness.codex_telemetry import CodexTelemetry
from agent_runtime.harness.trace_capture import TraceCapture
from reporting.native_timing import native_timing, timing_rows
from reporting.normalize import normalize
from reporting.pricing import Prices, Rate
from reporting.render import render
from reporting.schema import Step


def packet(name="codex.turn.ttft.duration_ms", *, count=1, total=100, start=1, end=2, temporality=1):
    """Model an actual OTLP/JSON histogram including an attribute to discard."""
    return {"resourceMetrics": [{"scopeMetrics": [{"metrics": [{"name": name,
        "histogram": {"aggregationTemporality": temporality, "dataPoints": [{
            "count": count, "sum": total, "startTimeUnixNano": str(start), "timeUnixNano": str(end),
            "attributes": [{"key": "model", "value": {"stringValue": "fixture"}},
                           {"key": "prompt", "value": {"stringValue": "PRIVATE"}}],
        }]}}]}]}]}


def test_loopback_roundtrip_and_isolation():
    """A real HTTP receipt belongs only to its invocation and retains no prompt."""
    with CodexTelemetry() as a, CodexTelemetry() as b:
        url = f"http://127.0.0.1:{a.server.server_port}{a.path}"
        with urlopen(Request(url, data=json.dumps(packet()).encode(),
                             headers={"Content-Type": "application/json"}), timeout=2) as response:
            assert response.status == 200
        assert a.evidence()["status"] == "captured"
        assert "PRIVATE" not in json.dumps(a.evidence())
        assert "prompt" not in json.dumps(a.evidence())
        assert b.evidence() == {"status": "unavailable", "samples": []}
        assert 'otel.metrics_exporter=' in a.arguments()[1]
    assert not a.thread.is_alive() and not b.thread.is_alive()


def test_histogram_temporality_and_duplicate_exports():
    """Re-exporting a point cannot double its duration; cumulative updates replace."""
    c = CodexTelemetry()
    c.ingest(packet()); c.ingest(packet())
    c.ingest(packet(total=50, start=2, end=3))
    assert sum(s["sum_ms"] for s in c.evidence()["samples"]) == 150
    c = CodexTelemetry()
    c.ingest(packet(temporality=2))
    c.ingest(packet(temporality=2, count=2, total=150, end=3))
    c.ingest(packet(temporality=2))
    assert len(c.evidence()["samples"]) == 1
    assert c.evidence()["samples"][0]["sum_ms"] == 150


def test_extra_diagnostics_are_not_retained():
    """The receiver persists only the first-token measurement used by reports."""
    c = CodexTelemetry()
    for name in ("codex.turn.e2e_duration_ms", "codex.responses_api_inference_time.duration_ms",
                 "codex.responses_api_engine_service_tbt.duration_ms"):
        c.ingest(packet(name))
    assert c.evidence() == {"status": "unavailable", "samples": []}
    c.ingest(packet())
    assert [sample["name"] for sample in c.evidence()["samples"]] == ["codex.turn.ttft.duration_ms"]


@pytest.mark.parametrize("total", [-1, float("nan"), float("inf")])
def test_invalid_duration_rejected(total):
    """Invalid measurements cannot become plausible timings."""
    with pytest.raises(ValueError):
        CodexTelemetry().ingest(packet(total=total))


def test_metadata_only_trace_roundtrip_and_render(tmp_path):
    """Native measurements survive normalization even when content is disabled."""
    telemetry = CodexTelemetry()
    telemetry.ingest(packet(total=60))
    model = FakeMessagesListChatModel(responses=[AIMessage(content="PRIVATE",
        response_metadata={"codex_telemetry": telemetry.evidence()},
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})],
        metadata={"ls_provider": "codex", "ls_model_name": "fixture"})
    capture = TraceCapture(tmp_path / "spans.jsonl", "codex", "fixture", capture_content=False)
    try:
        model.invoke([HumanMessage("PRIVATE")], config={"callbacks": [capture]})
    finally:
        capture.close()
    run = normalize(tmp_path / "spans.jsonl", title="Native timing")
    step = run.steps[0]
    assert not step.request and not step.response
    assert step.usage.output_tokens == 2
    assert native_timing(step)["ttft_ms"] == 60
    assert native_timing(step)["turn_ms"] == step.duration_ms
    assert "PRIVATE" not in run.model_dump_json()
    # Report rendering exposes telemetry even for a metadata-only recording.
    prices = Prices(as_of="2026-10-06", note="Fixture", models={"codex:fixture": Rate(input=1, output=2)})
    render(run, prices, tmp_path / "report.html")
    # Metadata-only views need not contain a response disclosure; check the filter projection too.
    assert timing_rows([step])[1]["ms"] == 60


def test_native_values_require_complete_successful_calls():
    """Multiple turns or partial capture cannot masquerade as a per-turn mean."""
    c = CodexTelemetry(); c.ingest(packet(total=50))
    step = Step(id="a", name="a", kind="model", provider="codex", start_ns=0,
                end_ns=200_000_000, status="ok", context={"codex_telemetry": json.dumps(c.evidence())})
    assert native_timing(step) == {"turn_ms": 200, "ttft_ms": 50}
    unknown = step.model_copy(update={"context": {}})
    assert all(row["ms"] is None for row in timing_rows([step, unknown]) if row["key"] != "turn_ms")
    assert timing_rows([step, unknown])[0]["ms"] == 200
    partial = timing_rows([step, unknown])[1]
    assert partial["observed_ms"] == 50 and partial["measured_calls"] == 1 and partial["total_calls"] == 2
    assert all(v is None for v in native_timing(step.model_copy(update={"status": "error"})).values())
    c.ingest(packet(start=2, end=3))
    step.context["codex_telemetry"] = json.dumps(c.evidence())
    assert native_timing(step)["ttft_ms"] is None
    step.context["codex_telemetry"] = "invalid"
    assert native_timing(step)["turn_ms"] == 200
    assert all(v is None for k, v in native_timing(step).items() if k != "turn_ms")
