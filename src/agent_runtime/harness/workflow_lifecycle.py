"""Share workflow context-audit policy and cleanup across console and HTTP lifetimes.

Callers choose when a conversation or HTTP turn finishes. These functions share
the evidence rules without owning either interface's execution loop or output.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json


def configure_context_audit(graph, *, capture_content, show_context=False):
    """Apply privacy settings before execution, including optional terminal output."""
    context = getattr(graph, "context_audit", None)
    if context:
        context.audit.capture_content = capture_content
        context.audit.show_context = show_context and capture_content


def save_context_audit(graph, directory):
    """Save available audit calls alongside the current run's report bundle.

    Report preparation clears stale artifacts. Workflows without audit calls
    produce no context file rather than retaining evidence from a previous run.
    """
    context = getattr(graph, "context_audit", None)
    if context and context.audit.calls:
        (directory / "context.json").write_text(
            json.dumps(context.evidence(), indent=2), encoding="utf-8"
        )


def close_run(graph):
    """Release session resources even if one resource fails to close.

    Detach resources before cleanup so repeated session teardown cannot close
    the same owner twice. Workspaces must still be removed if tracing shutdown
    raises; the cleanup error propagates to the interface that owns the session.
    """
    recorder = getattr(graph, "trace_recorder", None)
    workspace = getattr(graph, "workspace", None)
    if recorder is not None:
        graph.trace_recorder = None
    if workspace is not None:
        graph.workspace = None
    try:
        if recorder is not None:
            recorder.close()
    finally:
        if workspace is not None:
            workspace.cleanup()
