"""Run every local teaching sample and export standalone HTML and Excel reports.

Each sample owns a subdirectory so the default latest-run filenames cannot
collide. Reusing the batch directory refreshes generated files; --out can retain
a separate batch. No Langfuse or live model calls are launched. The existing
Excel exporter owns workbook formulas and presentation, not this orchestrator.
AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Explicit membership keeps hosted-tracing examples out and makes new lessons
# an intentional addition, rather than accidentally running arbitrary modules.
SAMPLES = (
    "simple_chat",
    "tool_chat",
    "subagent_chat",
    "thinking_agent",
    "review_loop",
    "expert_dispatch",
    "rag_chat",
    "mcp_rag_chat",
    "file_approval",
    "quote_request",
)


def excel_runtime():
    """Ensure the batch can produce Excel workbooks before starting any lessons.

    main calls this preflight so a missing exporter dependency fails before
    sample execution. Return the Node executable and child-process environment
    needed by run_sample. LG_EXCEL_RUNTIME overrides the desktop dependency
    directory; otherwise the bundled location is used. Missing Node or an
    unresolvable artifact-tool package raises RuntimeError.
    """
    bundled = (
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    )
    runtime = Path(os.environ.get("LG_EXCEL_RUNTIME", bundled)).expanduser().resolve()
    node = (
        str(runtime / "bin/node")
        if (runtime / "bin/node").is_file()
        else shutil.which("node")
    )
    if not node:
        raise RuntimeError(
            "Node.js is required for Excel export. Install Node.js or use the Codex desktop runtime."
        )
    env = {**os.environ, "LG_EXCEL_RUNTIME": str(runtime)}
    # Resolve the exporter package from the selected runtime, rather than the
    # current directory, to check the same dependency location used for export.
    probe = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            (
                "import {createRequire} from 'node:module'; import path from 'node:path'; "
                "createRequire(path.join(process.env.LG_EXCEL_RUNTIME,'package.json')).resolve('@oai/artifact-tool');"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        raise RuntimeError(
            "Excel runtime unavailable. Set LG_EXCEL_RUNTIME to the directory containing node_modules/@oai/artifact-tool."
        )
    return node, env


def run_sample(name, directory, *, prices, fx_file, node, env):
    """Produce one lesson's current HTML and Excel reports for the batch index.

    main calls this for each selected lesson so successful rows refer to this
    run's artifacts. name identifies a sample module; directory isolates its
    outputs. prices and optional fx_file supply accounting references. node and
    env come from excel_runtime and are passed to the exporter subprocess.

    Remove earlier report artifacts, run the scripted sample, convert run.json
    to workbook input, then export Excel. A failing child stops this sequence
    and raises to main; run.log retains its output. Success returns None after
    checking both reports are nonempty. Human decisions are scripted; the file
    lesson modifies its own batch output file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # Clear the prior result before launching children so failure cannot leave
    # an old workbook looking like the output of the current sample run.
    for filename in (
        "report.xlsx",
        "excel-data.json",
        "report.html",
        "run.json",
        "spans.jsonl",
        "prices.json",
        "turns-preview.png",
        "content-preview.png",
        "tree-preview.png",
        "reference-preview.png",
    ):
        (directory / filename).unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        f"samples.{name}.app",
        "--prices",
        str(prices),
        "--client",
        "static",
    ]
    if fx_file:
        command.extend(["--fx-file", str(fx_file)])
    if name == "file_approval":
        command.extend(
            [
                "--source",
                str(ROOT / "samples/file_approval/input.txt"),
                "--target",
                str(directory / "edited-summary.txt"),
                "--mode",
                "always-ask",
                "--decision",
                "approve",
            ]
        )
    # These stages depend on one another: the recorder writes run.json, the
    # converter extracts workbook data, and Node renders that data to Excel.
    commands = [
        command,
        [
            sys.executable,
            "-m",
            "lg_report.report.excel_data",
            "run.json",
            "--out",
            "excel-data.json",
        ],
        [
            node,
            str(ROOT / "src/lg_report/report/export_excel.mjs"),
            "excel-data.json",
            "report.xlsx",
        ],
    ]
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        for command in commands:
            subprocess.run(
                command,
                cwd=directory,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    # Successful exit alone is insufficient if an exporter stopped producing files.
    for filename in ("report.html", "report.xlsx"):
        if not (directory / filename).stat().st_size:
            raise RuntimeError(f"Empty output: {filename}")


def write_index(output, results):
    """Give the batch user a navigable view of completed lessons and failures.

    main calls this initially and after each lesson so the index reflects
    progress. output is the batch directory; results contains (sample_name,
    succeeded) pairs from this run. Replace index.html with links to successful
    reports and every lesson's log; filesystem errors propagate to the caller.
    """
    rows = []
    for name, ok in results:
        links = (
            (
                f'<a href="{name}/report.html">HTML</a> · '
                f'<a href="{name}/report.xlsx">Excel</a>'
            )
            if ok
            else "Failed"
        )
        rows.append(
            f'<li>{html.escape(name)} — {links} · <a href="{name}/run.log">Log</a></li>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Local sample reports</title>'
        "<style>body{font:18px system-ui;max-width:850px;margin:3rem auto;padding:0 1rem}li{margin:1rem 0}</style>"
        "<h1>Local sample reports</h1><p>Scripted models and human answers; RAG uses real local retrieval. "
        "No Langfuse or paid model calls. Each sample has its own standalone HTML and Excel report.</p>"
        "<ul>" + "".join(rows) + "</ul></html>",
        encoding="utf-8",
    )


def main():
    """Let one CLI invocation regenerate the local teaching report collection.

    The script entry point calls this to give the user one index of all lesson
    outcomes. Read CLI arguments, verify dependencies, then run samples in order
    and publish progress after each. A sample failure is recorded while later
    lessons still run. Return 0 only when every sample succeeds, otherwise 1;
    invalid arguments or preflight errors exit before the batch starts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/batch"),
        help="Batch directory, refreshed on rerun (default: ./reports/batch)",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=ROOT / "models.json",
        help="Shared pricing reference (default: repository models.json; no price fetch)",
    )
    parser.add_argument(
        "--fx-file",
        type=Path,
        help="Shared USD/EUR reference; otherwise use the daily cache/service",
    )
    args = parser.parse_args()
    try:
        node, env = excel_runtime()
        prices = args.prices.resolve(strict=True)
        fx_file = args.fx_file.resolve(strict=True) if args.fx_file else None
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    output = args.out.resolve()
    # Checked-in reference reports require deliberate review, not batch replacement.
    examples = ROOT / "reports/examples"
    if output == examples or examples in output.parents:
        parser.error("Choose a local output directory outside reports/examples")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    # Replace a previous batch's index immediately, so unfinished lessons are
    # not advertised as successes while this batch is still running.
    write_index(output, results)
    print(f"Batch output: {output}", flush=True)
    for name in SAMPLES:
        print(f"Running {name} ...", flush=True)
        try:
            run_sample(
                name, output / name, prices=prices, fx_file=fx_file, node=node, env=env
            )
            ok = True
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            ok = False
            print(
                f"  FAILED ({type(exc).__name__}); see {output / name / 'run.log'}",
                flush=True,
            )
        results.append((name, ok))
        write_index(output, results)
        if ok:
            print(f"  HTML + Excel: {output / name}", flush=True)
    print(
        f"{sum(ok for _, ok in results)}/{len(SAMPLES)} samples completed. Open {output / 'index.html'}"
    )
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
