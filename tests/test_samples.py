"""Run the documented sample commands from outside the repository directory.

Subprocess checks catch broken package imports, path assumptions, and missing
output artifacts that direct factory calls would miss. Explicit pricing and FX
files keep these tests offline; rerun checks protect prior evidence from overwrite.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lg_report.report.pricing import load_prices
from lg_report.report.render import conversation_turns
from lg_report.report.schema import Run


@pytest.mark.parametrize(
    "name,calls,tools,turn_count",
    [
        ("simple_chat", 2, 0, 2),
        ("tool_chat", 4, 2, 2),
        ("thinking_agent", 7, 6, 1),
        ("subagent_chat", 4, 2, 1),
        ("expert_dispatch", 12, 6, 3),
        ("review_loop", 4, 0, 1),
    ],
)
def test_standalone_application(name, calls, tools, turn_count, tmp_path):
    rate = tmp_path / "fx.json"
    rate.write_text(json.dumps({"rate": "0.871", "date": "2026-09-17"}))
    output = tmp_path / "report"
    # Run the actual script, not a factory imported from the reporting CLI.
    # A supplied rate file keeps this integration test fully offline.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"samples.{name}.app",
            "--prices",
            str(Path(__file__).resolve().parents[1] / "models.json"),
            "--fx-file",
            str(rate),
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert str(output / "report.html") in result.stdout
    run = Run.model_validate_json((output / "run.json").read_text())
    assert run.demo and run.status == "ok"
    assert sum(s.kind == "model" for s in run.steps) == calls
    assert sum(s.kind == "tool" for s in run.steps) == tools
    assert all(s.context.get("description") for s in run.steps)
    turns = conversation_turns(run, load_prices(output / "prices.json"))
    assert len(turns) == turn_count
    labels = [
        e["request_label"]
        for t in turns
        for e in t["events"]
        if e["step"].kind == "model"
    ]
    assert labels == [f"R{i + 1}" for i in range(calls)]
    # Reusing an output directory must not overwrite the original evidence.
    before = (output / "run.json").read_bytes()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            f"samples.{name}.app",
            "--prices",
            str(Path(__file__).resolve().parents[1] / "models.json"),
            "--fx-file",
            str(rate),
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        timeout=60,
    )
    assert again.returncode != 0
    assert (output / "run.json").read_bytes() == before
