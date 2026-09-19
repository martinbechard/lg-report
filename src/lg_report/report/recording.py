"""Run an invokable graph or conversation and save a complete local reporting bundle.

record_run attaches one tracing callback at the invocation boundary, then writes
raw spans, normalized run.json, the exact pricing snapshot, and an HTML report.
The graph itself therefore has no dependency on reporting. Nested operations
inherit callbacks so subagent work can be inspected in the same trace.

Cleanup attempts to save partial evidence even when execution fails; a report
file is not proof of success. Content capture requires explicit opt-in, output
directories must be new unless replacement is requested, and ambient hosted
LangSmith tracing is disabled here.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path
from uuid import uuid4

from langsmith import tracing_context

from lg_report.platform.cache_policy import CACHE_TTL
from lg_report.report.capture import TraceCapture
from lg_report.report.normalize import normalize
from lg_report.report.pricing import Prices
from lg_report.report.render import render
from lg_report.report.schema import Run


def record_run(
    agent,
    inputs,
    directory: Path,
    prices: Prices,
    *,
    provider: str,
    model: str,
    title="Chat agent run",
    demo=False,
    include_output=False,
    config=None,
    overwrite=False,
):
    """Make one execution inspectable and reproducible through a local report bundle.

    Call this at the application boundary to run an invokable agent/graph while
    retaining evidence for later analysis, including when execution fails.

    agent must support invoke(inputs, config=...); inputs/config follow that
    agent's contract. directory must be new, protecting prior run evidence from
    accidental overwrite unless overwrite=True explicitly permits replacement of
    the four report files in an existing directory. Other files are untouched.
    prices is the snapshot used by this run; provider/model
    are identity defaults when callback metadata is absent. include_output opts
    into saving message/tool content, which can contain sensitive information.

    Return the agent result unchanged. Invocation exceptions propagate after the
    finally block attempts export, so failures remain inspectable; I/O or export
    errors also propagate. The captured trace stays local even if the environment
    normally enables LangSmith tracing.
    """
    directory.mkdir(parents=True, exist_ok=overwrite)
    if overwrite:
        # Clear only this bundle, including derived files, so a failed export
        # cannot leave an old HTML report beside a newly captured trace.
        for name in ("spans.jsonl", "run.json", "prices.json", "report.html"):
            (directory / name).unlink(missing_ok=True)
    capture = TraceCapture(
        directory / "spans.jsonl", provider, model, capture_content=include_output
    )
    result = None
    status = "ok"
    try:
        # Callers may omit config/callbacks entirely; empty containers let us add
        # capture without discarding any supplied execution settings or handlers.
        run_config = {"recursion_limit": 30, **(config or {})}
        run_config["metadata"] = {
            **run_config.get("metadata", {}),
            "cache_ttl": CACHE_TTL,
        }
        run_config["callbacks"] = [*(run_config.get("callbacks") or []), capture]
        # This command owns local capture. Do not inherit ambient hosted tracing settings.
        with tracing_context(enabled=False):
            result = agent.invoke(inputs, config=run_config)
        # invoke executes now and returns the entire invocation result according
        # to this agent's contract. This is not an individual tool observation;
        # those are recorded separately by tool lifecycle callbacks. The pending
        # return below still runs finally before the caller receives this result.
        # Only dictionary graph state can carry the LangGraph interrupt marker.
        # A nonempty marker means execution paused for external input, even
        # though invoke returned normally; other result shapes remain successful.
        if isinstance(result, dict) and result.get("__interrupt__"):
            status = "interrupted"
        return result
    except BaseException:
        status = "error"
        raise
    finally:
        # Close unfinished spans before reading JSONL: failure evidence matters
        # most when the graph never reached its normal completion callbacks.
        capture.close()
        output = None
        # Saving content requires explicit opt-in. The result must also be graph
        # state (a dictionary) with at least one message before selecting its last
        # response as a run-level convenience snapshot, separate from the full
        # per-call conversation in spans. Failures before return leave result=None.
        # If any predicate fails, leave output absent rather than fabricate a
        # response or index an empty list.
        if include_output and isinstance(result, dict) and result.get("messages"):
            output = str(result["messages"][-1].content)
        # st_size is the file size in bytes: a nonzero size means JSONL evidence
        # is present for normalize to turn into steps.
        # A graph can fail before its first callback, leaving an empty file; then
        # create an empty report so the failure is still inspectable. Preserve an
        # invocation error, but classify any other evidence-free run as incomplete
        # rather than claim success without observed execution.
        run = (
            normalize(
                directory / "spans.jsonl",
                title=title,
                demo=demo,
                status=status,
                output=output,
            )
            if (directory / "spans.jsonl").stat().st_size
            else Run(
                id=str(uuid4()),
                title=title,
                demo=demo,
                status="error" if status == "error" else "incomplete",
                steps=[],
            )
        )
        (directory / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / "prices.json").write_text(
            prices.model_dump_json(indent=2), encoding="utf-8"
        )
        render(run, prices, directory / "report.html")
