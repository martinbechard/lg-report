"""Receive per-invocation Codex timing metrics on loopback without hosted services.

A separate receiver belongs to each CLI call, including concurrent calls, so
metrics cannot be attributed to another model or the judge. Only named timing
histograms are retained; prompts, paths, credentials, logs, and token charges
are not collected. CLI receipts remain the accounting authority. OTLP exports
flush when Codex exits; missing or malformed telemetry stays visibly unknown.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from uuid import uuid4

# The report needs only first-token telemetry. Complete turn duration and output
# rate come from the existing model-call span and usage receipt. Drop all other
# exported metrics before storing evidence.
TIMINGS = {"codex.turn.ttft.duration_ms"}


class CodexTelemetry:
    """Own a short-lived OTLP/JSON endpoint and retain bounded timing evidence."""

    def __init__(self):
        """Construct without binding a socket until the actual model invocation."""
        self.samples = {}
        self.invalid = False
        self.path = f"/{uuid4().hex}/v1/metrics"

    def ingest(self, packet):
        """Deduplicate exports and preserve temporality for accurate aggregation.

        Delta points sum distinct intervals. Cumulative points replace earlier
        observations of the same series, preventing periodic exports from
        counting the same turn twice. Internal series identity is not persisted.
        """
        for resource in packet.get("resourceMetrics", []):
            for scope in resource.get("scopeMetrics", []):
                for metric in scope.get("metrics", []):
                    name = metric.get("name")
                    if name not in TIMINGS:
                        continue
                    histogram = metric.get("histogram", {})
                    temporality = histogram.get("aggregationTemporality")
                    if temporality not in (1, 2):
                        raise ValueError("Unknown metric temporality")
                    for point in histogram.get("dataPoints", []):
                        count, total = int(point["count"]), float(point["sum"])
                        if count <= 0 or total < 0 or not math.isfinite(total):
                            raise ValueError("Invalid timing sample")
                        attrs = {a["key"]: a["value"] for a in point.get("attributes", [])}
                        start, end = int(point["startTimeUnixNano"]), int(point["timeUnixNano"])
                        key = (name, json.dumps(attrs, sort_keys=True), start,
                               end if temporality == 1 else None)
                        sample = {"name": name, "count": count, "sum_ms": total,
                                  "start_ns": start, "end_ns": end,
                                  "temporality": temporality}
                        for tag in ("app.version", "model"):
                            sample[tag] = attrs.get(tag, {}).get("stringValue")
                        if key not in self.samples or end >= self.samples[key]["end_ns"]:
                            self.samples[key] = sample

    def __enter__(self):
        """Listen only on loopback, with an unguessable per-call request path."""
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Accept bounded OTLP/JSON and suppress request logging/content."""

            def log_message(self, *_):
                """Keep the local endpoint and rejected payloads out of logs."""

            def do_POST(self):
                """Acknowledge valid metrics; mark malformed evidence explicitly."""
                self.connection.settimeout(2)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if self.path != owner.path or not 0 < length <= 2_000_000:
                        self.send_error(400)
                        return
                    owner.ingest(json.loads(self.rfile.read(length)))
                except (ValueError, KeyError, TypeError, AttributeError, OSError):
                    owner.invalid = True
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        return self

    def arguments(self):
        """Pass settings explicitly because the adapter ignores user config."""
        endpoint = f"http://127.0.0.1:{self.server.server_port}{self.path}"
        return ["-c", 'otel.metrics_exporter={otlp-http={endpoint="' + endpoint + '",protocol="json"}}']

    def evidence(self):
        """Return sanitized native measurements after the CLI has flushed."""
        return {"status": "invalid" if self.invalid else "captured" if self.samples else "unavailable",
                "samples": list(self.samples.values())}

    def __exit__(self, *_):
        """Reap the local receiver on success, provider error, or cancellation."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
