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
# Run each documented sample from outside the repository to catch
# packaging, path, artifact, and rerun regressions that direct imports miss.
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


def test_working_directory_defaults_and_report_commands(tmp_path):
    """Exercise the copy/paste workflow and replacement without mixing two runs.

    Launch outside the checkout to distinguish cwd from source-relative paths.
    Keep unrelated files to prove replacement is confined to the report bundle.
    """
    rate = tmp_path / "fx.json"
    rate.write_text(json.dumps({"rate": "0.871", "date": "2026-09-17"}))
    sentinel = tmp_path / "notes.txt"
    sentinel.write_text("keep me")
    catalog = Path(__file__).resolve().parents[1] / "models.json"

    def execute(*args):
        # Run a documented Python command as a user outside the checkout would.
        # args supplies interpreter arguments. Fail with captured stderr on a nonzero
        # exit; otherwise return stdout for path/report assertions.
        result = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    ids = []
    for sample, calls in [("simple_chat", 2), ("tool_chat", 4)]:
        stdout = execute(
            "-m",
            f"samples.{sample}.app",
            "--prices",
            str(catalog),
            "--fx-file",
            str(rate),
        )
        assert str(tmp_path / "report.html") in stdout
        run = Run.model_validate_json((tmp_path / "run.json").read_text())
        ids.append(run.id)
        assert sum(step.kind == "model" for step in run.steps) == calls
    assert ids[0] != ids[1]
    assert sentinel.read_text() == "keep me"
    assert rate.exists()
    for name in ("report.html", "run.json", "spans.jsonl", "prices.json"):
        assert (tmp_path / name).stat().st_size > 0

    # The no-argument commands consume the latest bundle. Explicit input paths
    # place derived outputs beside that input; explicit --out still wins.
    (tmp_path / "run.json").unlink()
    execute("-m", "lg_report", "normalize", "--demo")
    rebuilt = Run.model_validate_json((tmp_path / "run.json").read_text())
    assert rebuilt.demo
    assert sum(step.kind == "model" for step in rebuilt.steps) == 4
    (tmp_path / "report.html").unlink()
    execute("-m", "lg_report", "--fx-file", str(rate), "render")
    assert (tmp_path / "report.html").stat().st_size > 0
    saved = tmp_path / "saved"
    saved.mkdir()
    for name in ("spans.jsonl", "prices.json"):
        (saved / name).write_bytes((tmp_path / name).read_bytes())
    execute("-m", "lg_report", "normalize", "saved/spans.jsonl", "--demo")
    execute("-m", "lg_report", "--fx-file", str(rate), "render", "saved/run.json")
    assert (saved / "report.html").stat().st_size > 0
    execute("-m", "lg_report", "normalize", "--out", "rebuilt.json")
    assert (tmp_path / "rebuilt.json").exists()
    execute("-m", "lg_report", "--fx-file", str(rate), "render", "--out", "other.html")
    assert (tmp_path / "other.html").exists()


def test_batch_includes_all_local_samples():
    """A new local lesson must join the batch; hosted tracing must stay excluded."""
    import runpy

    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts/run_samples.py"))
    local = {
        p.parent.name
        for p in (root / "samples").glob("*/app.py")
        if "langfuse" not in p.parent.name
    }
    assert set(runner["SAMPLES"]) == local


def test_batch_continues_after_failure_and_hides_stale_links(tmp_path, monkeypatch):
    """Publish failure diagnostics without presenting a prior Excel as current."""
    import runpy

    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts/run_samples.py"))
    main = runner["main"]
    state = main.__globals__
    monkeypatch.setitem(state, "excel_runtime", lambda: ("node", {}))
    monkeypatch.setitem(state, "SAMPLES", ("simple_chat", "tool_chat"))
    calls = []

    def fake_sample(name, directory, **kwargs):
        # Force one sample to fail so the batch must still attempt the next sample.
        # Record its name; directory and kwargs are unused because no report is built.
        calls.append(name)
        if name == "simple_chat":
            raise subprocess.CalledProcessError(1, "sample")

    monkeypatch.setitem(state, "run_sample", fake_sample)
    monkeypatch.setattr(sys, "argv", ["run_samples.py", "--out", str(tmp_path)])
    assert main() == 1
    assert calls == ["simple_chat", "tool_chat"]
    index = (tmp_path / "index.html").read_text()
    assert "simple_chat/report.xlsx" not in index
    assert "simple_chat/run.log" in index
    assert "tool_chat/report.xlsx" in index


def test_batch_clears_generated_files_before_failed_sample(tmp_path, monkeypatch):
    """A failed refresh must not leave stale reports looking like new results."""
    import runpy

    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts/run_samples.py"))
    for filename in ("report.xlsx", "report.html", "run.json", "notes.txt"):
        (tmp_path / filename).write_text("old")

    def fail(command, **kwargs):
        # Fail the subprocess after checking batch execution cannot wait for input.
        # command is the attempted child invocation; kwargs exposes its stdin setting.
        # The raised error lets the caller test stale-output cleanup.
        assert kwargs["stdin"] == subprocess.DEVNULL
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner["run_sample"](
            "simple_chat",
            tmp_path,
            prices=root / "models.json",
            fx_file=None,
            node="node",
            env={},
        )
    assert not (tmp_path / "report.xlsx").exists()
    assert not (tmp_path / "report.html").exists()
    assert not (tmp_path / "run.json").exists()
    assert (tmp_path / "notes.txt").read_text() == "old"
    assert (tmp_path / "run.log").exists()
