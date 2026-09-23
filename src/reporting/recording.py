"""Export captured execution evidence as a local reporting bundle.

save_report normalizes closed spans and writes run.json, the pricing snapshot,
and HTML. clear_report removes only generated artifacts before replacement.
Console execution and streamed web runs share these functions; this module does
not execute workflows or own capture lifetime. Empty evidence never proves success.
Architecture and ownership: docs/chat-composition.md.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path
from uuid import uuid4

from reporting.normalize import normalize
from reporting.render import render
from reporting.schema import Run

# Only generated artifacts belong to replacement; user inputs beside them stay safe.
REPORT_FILES = (
    "spans.jsonl",
    "run.json",
    "prices.json",
    "report.html",
    "context.json",
    "report.xlsx",
    "excel-data.json",
    "turns-preview.png",
    "content-preview.png",
    "tree-preview.png",
    "reference-preview.png",
)


def clear_report(directory: Path):
    """Remove the previous generated bundle before publishing a replacement.

    Samples and browser turns share this list so stale Excel/context artifacts
    cannot survive beside new HTML. Never delete arbitrary files or directories:
    an explicit output path can also contain user inputs or an open batch log.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name in REPORT_FILES:
        (directory / name).unlink(missing_ok=True)


def save_report(
    directory, prices, *, title, demo=False, status="ok", output=None, run_id=None
):
    """Export closed trace evidence using the same accounting for every interface.

    The caller owns capture lifetime and publication timing. Console runs export
    in place; HTTP runs export privately before publishing the completed bundle.
    An empty trace is incomplete unless execution failed, so absence of evidence
    never becomes a successful run. output must already obey content-capture policy.
    """
    source = directory / "spans.jsonl"
    run = (
        normalize(source, title=title, demo=demo, status=status, output=output)
        if source.stat().st_size
        else Run(
            id=run_id or str(uuid4()),
            title=title,
            demo=demo,
            status="error" if status == "error" else "incomplete",
            steps=[],
        )
    )
    (directory / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    (directory / "prices.json").write_text(
        prices.model_dump_json(indent=2), encoding="utf-8"
    )
    render(run, prices, directory / "report.html")
    return run
