"""Project provider timing receipts into shared report measurements.

The collector stores sanitized OTLP histograms on each model step. This reader
accepts only complete, finite timing evidence and preserves absence as unknown.
Metric timestamps mark histogram collection, not the start/end of inference.
Direct API calls use client elapsed time and the first streamed output callback;
native telemetry supplies internal measurements where the provider exposes them.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import math

LABELS = {"turn_ms": "Turn time", "ttft_ms": "Time to first token"}


def native_timing(step):
    """Project complete call time and first-token evidence without diagnostics.

    Turn time uses the full client call for every provider. Codex first-token
    telemetry requires one native observation; API first output is observed by
    the streaming callback. The report explains these different TTFT boundaries.
    """
    result = dict.fromkeys(LABELS)
    if step.status != "ok" or step.kind != "model":
        return result
    result["turn_ms"] = step.duration_ms
    if step.provider in {"openai", "anthropic"}:
        # The client can time the request and first output, but cannot infer
        # backend inference/overhead or wrapper time from those two timestamps.
        first = step.context.get("client_ttft_ns")
        if type(first) is int and 0 <= first / 1_000_000 <= step.duration_ms:
            result["ttft_ms"] = first / 1_000_000
        return result
    if step.provider != "codex":
        return result
    try:
        evidence = json.loads(step.context.get("codex_telemetry", "{}"))
        if evidence.get("status") != "captured":
            return result
        samples = [sample for sample in evidence["samples"]
                   if sample["name"] == "codex.turn.ttft.duration_ms"]
        # A per-invocation first-token time needs exactly one turn, not the sum
        # or average of an unknown number of internal turns.
        if len(samples) == 1:
            sample = samples[0]
            if (type(sample["count"]) is int and sample["count"] == 1
                    and math.isfinite(sample["sum_ms"]) and sample["sum_ms"] >= 0):
                result["ttft_ms"] = sample["sum_ms"]
    except (ValueError, KeyError, TypeError, AttributeError):
        # Malformed optional telemetry does not erase the measured client call.
        return {**dict.fromkeys(LABELS), "turn_ms": step.duration_ms}
    return result


def timing_rows(calls):
    """Keep complete means distinct from measurements covering only some calls.

    Partial native capture cannot stand in for a whole execution. Complete means
    use every call; an observed mean is accompanied by coverage for diagnostic
    display. Judge and user calls are excluded by the comparison caller.
    """
    values = [native_timing(call) for call in calls]
    rows = []
    for key, label in LABELS.items():
        observed = [value[key] for value in values if value[key] is not None]
        mean = sum(observed) / len(observed) if observed else None
        rows.append({"key": key, "label": label, "ms": mean if len(observed) == len(values) else None,
                     "observed_ms": mean, "measured_calls": len(observed), "total_calls": len(values)})
    return rows
