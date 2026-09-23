"""Convert this application's OTel JSONL evidence into report schema v1.

AI attribution: Generated with AI assistance.

This adapter expects the lg.* attributes written by TraceCapture, not arbitrary
OTel exports. Provider SDK usage has already reached LangChain's common shape;
this boundary gives every exporter the same inclusive totals and cache subsets.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from datetime import datetime
from pathlib import Path

from reporting.schema import Run, Step, Usage


def timestamp_ns(value: str) -> int:
    """Give report steps comparable clock values for ordering and elapsed time.

    ``value`` is an exported ISO timestamp. Return Unix nanoseconds limited to
    datetime's microsecond precision; malformed timestamps raise ValueError."""
    timestamp = datetime.fromisoformat(value)
    return int(timestamp.timestamp()) * 1_000_000_000 + timestamp.microsecond * 1_000


def normalize(
    path: Path, *, title: str, demo: bool = False, status="ok", output=None
) -> Run:
    """Prepare captured execution evidence for every exporter to interpret consistently.

    ``path`` is the completed JSONL file from TraceCapture. Return a validated
    Run containing chronologically ordered Steps and inclusive Usage records;
    pricing is a later operation, so this conversion requires no network calls.

    title/demo/status/output describe the invocation supplied by its recorder.
    A supplied failure status is retained; an apparently successful invocation
    is downgraded when a root span reports failure, interruption, or incompletion.
    Malformed JSON, missing required attributes, and invalid schema data fail
    rather than silently dropping operations and undercounting a run.
    """
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        exported_span = json.loads(line)
        attributes = exported_span["attributes"]
        usage = None
        # Only an explicit SDK usage record establishes billable token counts;
        # without it the step remains unmetered, not a zero-token call.
        if "lg.usage" in attributes:
            sdk_usage = json.loads(attributes["lg.usage"])
            input_details = sdk_usage.get("input_token_details", {})
            output_details = sdk_usage.get("output_token_details", {})
            # Optional lifetime counters may be absent or None even when total
            # usage exists. They then establish no explicit lifetime allocation;
            # the separate cache_creation total below still preserves writes.
            write_5m = input_details.get("ephemeral_5m_input_tokens", 0) or 0
            write_1h = input_details.get("ephemeral_1h_input_tokens", 0) or 0
            # Some SDKs supply both a write total and TTL breakdown. They describe
            # the same tokens; adding them would charge cache creation twice.
            usage = Usage(
                input_tokens=sdk_usage["input_tokens"],
                output_tokens=sdk_usage["output_tokens"],
                cache_read=input_details.get("cache_read", 0),
                cache_write=max(
                    input_details.get("cache_creation", 0) or 0, write_5m + write_1h
                ),
                cache_write_5m=write_5m,
                cache_write_1h=write_1h,
                reasoning=output_details.get("reasoning", 0),
            )
        steps.append(
            Step(
                id=attributes["lg.run_id"],
                parent_id=attributes.get("lg.parent_run_id"),
                name=exported_span["name"],
                kind=attributes["lg.kind"],
                start_ns=timestamp_ns(exported_span["start_time"]),
                end_ns=timestamp_ns(exported_span["end_time"]),
                # An unclosed span lacks completion evidence even if other flags
                # exist. Otherwise an explicit approval pause outranks SDK error
                # status; GraphInterrupt is control flow. Only an ERROR status
                # without either flag is failure; remaining completed spans are OK.
                status="incomplete"
                if attributes.get("lg.incomplete")
                else "interrupted"
                if attributes.get("lg.interrupted")
                else "error"
                if exported_span["status"]["status_code"] == "ERROR"
                else "ok",
                provider=attributes.get("gen_ai.provider.name"),
                model=attributes.get(
                    "gen_ai.response.model", attributes.get("gen_ai.request.model")
                ),
                effort=attributes.get("lg.effort"),
                usage=usage,
                error=attributes.get("lg.error"),
                request=json.loads(attributes.get("lg.request", "[]")),
                response=json.loads(attributes.get("lg.response", "[]")),
                tool_definitions=json.loads(
                    attributes.get("lg.tool_definitions", "null")
                ),
                context=json.loads(attributes.get("lg.context", "{}")),
            )
        )
    # No observed operations means there is no trace to normalize; the recorder
    # handles evidence-free invocations separately with an empty Run.
    if not steps:
        raise ValueError("No spans found in the trace")
    # JSONL is emitted when spans END, so children usually precede their parents.
    # Restore execution order; equal starts put the enclosing operation first.
    steps.sort(key=lambda s: (s.start_ns, -s.end_ns))
    # Only spans without a parent represent whole invocations. Child failures
    # may be handled by their parent and must not independently fail the run.
    roots = [s for s in steps if s.parent_id is None]
    # Without a root there is no invocation identity or trustworthy run status.
    if not roots:
        raise ValueError("Trace contains no root span")
    # Preserve any caller-supplied error/pause/incomplete status. Downgrade only
    # an apparently successful invocation whose root evidence says otherwise.
    # Across multiple roots, missing completion wins, then an approval pause,
    # then error; this identifies the unfinished work before completed failures.
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
