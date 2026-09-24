"""Execute a synchronous runnable with local trace capture and report export.

The runnable supplies invoke(inputs, config=...). A graph executes one invocation;
Conversation drives its whole client loop. This wrapper invokes either once and
exports evidence through reporting.recording, including on execution failure.
Ambient LangSmith tracing is disabled; explicitly supplied callbacks remain active.
Architecture and ownership: docs/chat-composition.md.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path

from langsmith import tracing_context

from agent_runtime.harness.cache_policy import CACHE_TTL
from agent_runtime.harness.trace_capture import TraceCapture
from agent_runtime.harness.workflow_diagram import diagram_config
from reporting.pricing import Prices
from reporting.recording import clear_report, save_report


def execute_runnable(
    runnable,
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
    """Invoke a runnable synchronously and save its execution evidence locally.

    runnable is an object with invoke(inputs, config=...), such as a compiled
    graph or the harness Conversation. This function executes it, not merely
    observes it. A Conversation invocation includes all client turns and resumes;
    a direct graph invocation can return while waiting for an interruption answer.

    inputs is passed unchanged to invoke. config supplies execution settings and
    existing callbacks; local capture is appended without replacing them. The
    runnable must propagate those callbacks for its work to appear in the trace.
    directory must be new, protecting prior run evidence from
    accidental overwrite unless overwrite=True explicitly permits replacement of
    the generated report bundle in an existing directory. Other files are untouched.
    prices is the snapshot used by this run; provider/model
    are identity defaults when callback metadata is absent. include_output opts
    into saving message/tool content, which can contain sensitive information.

    Return the invocation result unchanged after saving the report. Execution is
    not retried. Invocation exceptions propagate after the
    finally block attempts export, so failures remain inspectable; I/O or export
    errors also propagate. The captured trace stays local even if the environment
    normally enables LangSmith tracing.
    """
    directory.mkdir(parents=True, exist_ok=overwrite)
    if overwrite:
        # Clear only this bundle, including derived files, so a failed export
        # cannot leave an old HTML report beside a newly captured trace.
        clear_report(directory)
    capture = TraceCapture(
        directory / "spans.jsonl", provider, model, capture_content=include_output
    )
    result = None
    status = "ok"
    try:
        # Callers may omit config/callbacks entirely; empty containers let us add
        # capture without discarding any supplied execution settings or handlers.
        run_config = diagram_config(runnable, {"recursion_limit": 30, **(config or {})})
        run_config["metadata"] = {
            **run_config.get("metadata", {}),
            "cache_ttl": CACHE_TTL,
        }
        run_config["callbacks"] = [*(run_config.get("callbacks") or []), capture]
        # Execute once with local capture attached. Explicit callbacks such as
        # Langfuse remain active; only ambient LangSmith tracing is disabled.
        with tracing_context(enabled=False):
            result = runnable.invoke(inputs, config=run_config)
        # A direct graph invocation can return an interruption in its state.
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
        save_report(
            directory, prices, title=title, demo=demo, status=status, output=output
        )
