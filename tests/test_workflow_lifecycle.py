"""Verify shared cleanup and evidence rules independently of interface timing.

A failed exporter must not strand a workflow workspace, and an empty trace must
not be presented as a successful execution in either console or web reports.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from datetime import date
from types import SimpleNamespace

import pytest

from agent_runtime.harness.workflow_lifecycle import (
    close_run,
    configure_context_audit,
    save_context_audit,
)
from reporting.pricing import Prices
from reporting.recording import save_report


def test_cleanup_releases_workspace_after_recorder_failure():
    """Tracing failure cannot leak workspace files or trigger double shutdown."""
    closed = []

    def fail_close():
        closed.append("recorder")
        raise RuntimeError("shutdown failed")

    graph = SimpleNamespace(
        trace_recorder=SimpleNamespace(close=fail_close),
        workspace=SimpleNamespace(cleanup=lambda: closed.append("workspace")),
    )
    with pytest.raises(RuntimeError, match="shutdown failed"):
        close_run(graph)
    close_run(graph)
    assert closed == ["recorder", "workspace"]


@pytest.mark.parametrize("capture_content", [False, True])
def test_audit_policy_controls_terminal_and_saved_content(tmp_path, capture_content):
    """Shared setup applies privacy before audit evidence is collected or displayed."""
    audit = SimpleNamespace(calls=[1], capture_content=None, show_context=None)
    context = SimpleNamespace(
        audit=audit,
        evidence=lambda: {"content": "prompt" if audit.capture_content else None},
    )
    graph = SimpleNamespace(context_audit=context)
    configure_context_audit(graph, capture_content=capture_content, show_context=True)
    assert audit.show_context is capture_content
    save_context_audit(graph, tmp_path)
    import json

    evidence = json.loads((tmp_path / "context.json").read_text())
    assert evidence["content"] == ("prompt" if capture_content else None)


@pytest.mark.parametrize("status,expected", [("ok", "incomplete"), ("error", "error")])
def test_empty_trace_exports_truthful_report(tmp_path, status, expected):
    """Both interfaces must retain failure evidence without fabricating success."""
    (tmp_path / "spans.jsonl").touch()
    prices = Prices(as_of=date(2026, 9, 22), note="Test fixture", models={})
    run = save_report(
        tmp_path, prices, title="Empty run", status=status, run_id="empty-run"
    )
    assert run.id == "empty-run"
    assert run.status == expected
    assert run.steps == []
    for name in ("run.json", "prices.json", "report.html"):
        assert (tmp_path / name).is_file()
