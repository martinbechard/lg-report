"""Run the documented sample commands from outside the repository directory.

Subprocess checks catch broken package imports, path assumptions, and missing
output artifacts that direct factory calls would miss. Explicit pricing and FX
files keep these tests offline; rerun checks ensure old generated output is replaced.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reporting.pricing import load_prices
from reporting.render import conversation_turns
from reporting.schema import Run


@pytest.mark.parametrize(
    "name,calls,tools,turn_count",
    [
        ("simple_chat", 2, 0, 2),
        ("tool_chat", 4, 2, 2),
        ("shell_script", 2, 1, 1),
        ("thinking_agent", 7, 6, 1),
        ("subagent_chat", 4, 2, 1),
        ("context_budget", 17, 6, 2),
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
            "agent_runtime",
            "--sample",
            name,
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
    # Reusing a sample output must replace evidence and discard stale exports.
    (output / "report.xlsx").write_text("stale workbook")
    before = (output / "run.json").read_bytes()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "--sample",
            name,
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
    assert again.returncode == 0, again.stderr
    assert (output / "run.json").read_bytes() != before
    assert not (output / "report.xlsx").exists()


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

    working = tmp_path

    def execute(*args):
        # Run a documented Python command as a user outside the checkout would.
        # args supplies interpreter arguments. Fail with captured stderr on a nonzero
        # exit; otherwise return stdout for path/report assertions.
        result = subprocess.run(
            [sys.executable, *args],
            cwd=working,
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
            "agent_runtime",
            "--sample",
            sample,
            "--prices",
            str(catalog),
            "--fx-file",
            str(rate),
        )
        output = tmp_path / "reports" / sample
        assert str(output / "report.html") in stdout
        run = Run.model_validate_json((output / "run.json").read_text())
        ids.append(run.id)
        assert sum(step.kind == "model" for step in run.steps) == calls
    assert ids[0] != ids[1]
    assert sentinel.read_text() == "keep me"
    assert rate.exists()
    working = output
    for name in ("report.html", "run.json", "spans.jsonl", "prices.json"):
        assert (output / name).stat().st_size > 0

    # The no-argument commands consume the latest bundle. Explicit input paths
    # place derived outputs beside that input; explicit --out still wins.
    (output / "run.json").unlink()
    execute("-m", "reporting", "normalize", "--demo")
    rebuilt = Run.model_validate_json((output / "run.json").read_text())
    assert rebuilt.demo
    assert sum(step.kind == "model" for step in rebuilt.steps) == 4
    (output / "report.html").unlink()
    execute("-m", "reporting", "--fx-file", str(rate), "render")
    assert (output / "report.html").stat().st_size > 0
    saved = output / "saved"
    saved.mkdir()
    for name in ("spans.jsonl", "prices.json"):
        (saved / name).write_bytes((output / name).read_bytes())
    execute("-m", "reporting", "normalize", "saved/spans.jsonl", "--demo")
    execute("-m", "reporting", "--fx-file", str(rate), "render", "saved/run.json")
    assert (saved / "report.html").stat().st_size > 0
    execute("-m", "reporting", "normalize", "--out", "rebuilt.json")
    assert (output / "rebuilt.json").exists()
    execute("-m", "reporting", "--fx-file", str(rate), "render", "--out", "other.html")
    assert (output / "other.html").exists()


def test_batch_includes_all_local_samples():
    """A new local lesson must join the batch; hosted tracing must stay excluded."""
    import runpy

    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts/run_samples.py"))
    import json

    local = {
        entry["id"]
        for path in (root / "samples").glob("*/sample.json")
        for entry in json.loads(path.read_text())["samples"]
        if entry.get("tracing", "local") == "local"
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
    refreshes = []

    def failed_refresh(command, **kwargs):
        """A failed FX refresh must not prevent sample execution."""
        refreshes.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", failed_refresh)
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
    assert len(refreshes) == 1
    assert refreshes[0][1].endswith("scripts/update_exchange_rate.py")
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


@pytest.mark.parametrize("simulated", [False, True])
def test_batch_model_mode_and_shared_config(tmp_path, monkeypatch, simulated):
    """The default must reach real providers; offline execution requires opt-in.

    Replace child execution so this contract test never bills a provider. Check
    shell precedence and index wording as well as the mode sent to each lesson.
    """
    import runpy

    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/run_samples.py"))["main"]
    state = main.__globals__
    config = tmp_path / "config.env"
    config.write_text(
        "LG_PROVIDER=openai\nLG_MODEL=file-model\nOPENAI_API_KEY=test-placeholder\n"
    )
    monkeypatch.setitem(
        state, "excel_runtime", lambda: ("node", {"LG_MODEL": "shell-model"})
    )
    monkeypatch.setitem(state, "SAMPLES", ("simple_chat",))
    refreshes = []

    def refresh(command, **kwargs):
        """Keep mode/configuration tests independent of live exchange services."""
        refreshes.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", refresh)
    calls = []

    def capture(name, directory, **kwargs):
        """Capture the resolved mode/configuration instead of invoking a model."""
        calls.append(kwargs)

    monkeypatch.setitem(state, "run_sample", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_samples.py",
            "--out",
            str(tmp_path),
            "--env-file",
            str(config),
            *(["--simulated"] if simulated else []),
        ],
    )
    assert main() == 0
    assert len(refreshes) == (0 if simulated else 1)
    assert calls[0]["fx_file"] == root / "exchange-rate.json"
    assert calls[0]["simulated"] is simulated
    assert calls[0]["env"]["LG_MODEL"] == "shell-model"
    index = (tmp_path / "index.html").read_text()
    assert ("Simulated models" if simulated else "Real provider models") in index
    assert ("no paid model calls" in index) is simulated


@pytest.mark.parametrize("simulated", [False, True])
@pytest.mark.parametrize("sample", ["simple_chat", "quote_request"])
def test_batch_child_model_and_interaction_mode(
    tmp_path, monkeypatch, simulated, sample
):
    """Real calls pass --live; live quote questions retain terminal input/output."""
    import runpy

    root = Path(__file__).resolve().parents[1]
    run_sample = runpy.run_path(str(root / "scripts/run_samples.py"))["run_sample"]
    calls = []

    def child(command, **kwargs):
        """Simulate completed artifacts while recording actual subprocess options."""
        calls.append((command, kwargs))
        for filename in ("report.html", "report.xlsx"):
            (tmp_path / filename).write_text("generated")

    monkeypatch.setattr(subprocess, "run", child)
    run_sample(
        sample,
        tmp_path,
        prices=root / "models.json",
        fx_file=None,
        node="node",
        env={},
        simulated=simulated,
    )
    command, options = calls[0]
    assert ("--live" in command) is not simulated
    interactive = sample == "quote_request" and not simulated
    assert command[command.index("--client") + 1] == (
        "console" if interactive else "static"
    )
    assert options["stdin"] == (None if interactive else subprocess.DEVNULL)
    assert (options["stdout"] is None) is interactive
    assert all(options["stdin"] == subprocess.DEVNULL for _, options in calls[1:])


def test_live_luna_uses_responses_api(monkeypatch):
    """Luna tool calls need Responses when reasoning is enabled by the provider.

    Construct the real adapter without invoking it, so this regression check
    verifies endpoint selection without credentials or a billed model request.
    """
    from agent_runtime.harness.model_config import configured_model

    monkeypatch.setenv("LG_PROVIDER", "openai")
    monkeypatch.setenv("LG_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.delenv("LG_EFFORT", raising=False)
    adapter, provider, model = configured_model()
    assert adapter.use_responses_api is True
    assert (provider, model) == ("openai", "gpt-5.6-luna")


@pytest.mark.parametrize("available", [False, True])
def test_sample_settings_never_fetch_exchange_rate(tmp_path, monkeypatch, available):
    """A missing FX file leaves EUR unknown without network or model-start delays."""
    import urllib.request

    from agent_runtime.harness.argument_parser import argument_parser
    from agent_runtime.harness.settings import settings_for

    root = Path(__file__).resolve().parents[1]
    app = tmp_path / "samples/simple_chat/sample.json"
    rate = tmp_path / "exchange-rate.json"
    if available:
        rate.write_text('{"rate":"0.88","date":"2000-01-01"}')
    monkeypatch.delenv("LG_FX_FILE", raising=False)

    def forbidden(*args, **kwargs):
        """Fail if settings tries to repair absent FX using a network request."""
        pytest.fail("Individual samples must never fetch exchange rates")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    args = argument_parser(str(app), "FX boundary test").parse_args(
        ["--prices", str(root / "models.json")]
    )
    settings = settings_for(str(app), "FX boundary test", args=args)
    assert (settings.prices.exchange is not None) is available
    if not available:
        assert "FileNotFoundError" in settings.prices.exchange_error
