"""Convert this application's OTel SDK JSONL export into report schema v1."""

import json
from datetime import datetime
from pathlib import Path

from .schema import Run, Step, Usage


def timestamp_ns(value: str) -> int:
    dt = datetime.fromisoformat(value)
    return int(dt.timestamp()) * 1_000_000_000 + dt.microsecond * 1_000


def normalize(
    path: Path, *, title: str, demo: bool = False, status="ok", output=None
) -> Run:
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        attrs = raw["attributes"]
        usage = None
        if "lg.usage" in attrs:
            u = json.loads(attrs["lg.usage"])
            inputs = u.get("input_token_details", {})
            outputs = u.get("output_token_details", {})
            write_5m = inputs.get("ephemeral_5m_input_tokens", 0) or 0
            write_1h = inputs.get("ephemeral_1h_input_tokens", 0) or 0
            usage = Usage(
                input_tokens=u["input_tokens"],
                output_tokens=u["output_tokens"],
                cache_read=inputs.get("cache_read", 0),
                cache_write=max(
                    inputs.get("cache_creation", 0) or 0, write_5m + write_1h
                ),
                cache_write_5m=write_5m,
                cache_write_1h=write_1h,
                reasoning=outputs.get("reasoning", 0),
            )
        steps.append(
            Step(
                id=attrs["lg.run_id"],
                parent_id=attrs.get("lg.parent_run_id"),
                name=raw["name"],
                kind=attrs["lg.kind"],
                start_ns=timestamp_ns(raw["start_time"]),
                end_ns=timestamp_ns(raw["end_time"]),
                status="incomplete"
                if attrs.get("lg.incomplete")
                else "interrupted"
                if attrs.get("lg.interrupted")
                else "error"
                if raw["status"]["status_code"] == "ERROR"
                else "ok",
                provider=attrs.get("gen_ai.provider.name"),
                model=attrs.get(
                    "gen_ai.response.model", attrs.get("gen_ai.request.model")
                ),
                effort=attrs.get("lg.effort"),
                usage=usage,
                error=attrs.get("lg.error"),
                request=json.loads(attrs.get("lg.request", "[]")),
                response=json.loads(attrs.get("lg.response", "[]")),
                context=json.loads(attrs.get("lg.context", "{}")),
            )
        )
    if not steps:
        raise ValueError("No spans found in the trace")
    steps.sort(key=lambda s: (s.start_ns, -s.end_ns))
    roots = [s for s in steps if s.parent_id is None]
    if not roots:
        raise ValueError("Trace contains no root span")
    if status == "ok" and any(s.status != "ok" for s in roots):
        status = (
            "incomplete"
            if any(s.status == "incomplete" for s in roots)
            else "interrupted"
            if any(s.status == "interrupted" for s in roots)
            else "error"
        )
    return Run(
        id=roots[0].id,
        title=title,
        demo=demo,
        status=status,
        steps=steps,
        output=output,
    )
