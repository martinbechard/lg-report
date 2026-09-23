"""Run every local teaching sample and export standalone HTML and Excel reports.

Each sample owns a subdirectory so the default latest-run filenames cannot
collide. Reusing the batch directory refreshes generated files; --out can retain
a separate batch. Real provider models run by default; --simulated explicitly
selects offline models. Langfuse examples are excluded. The existing
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

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
from agent_runtime.harness.sample_catalog import SampleCatalog

SAMPLES = tuple(
    sample.id
    for sample in SampleCatalog().samples.values()
    if sample.tracing == "local"
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


def run_sample(name, directory, *, prices, fx_file, node, env, simulated=False):
    """Produce one lesson's current HTML and Excel reports for the batch index.

    main calls this for each selected lesson so successful rows refer to this
    run's artifacts. name identifies a sample module; directory isolates its
    outputs. prices and optional fx_file supply accounting references. node and
    env come from excel_runtime and are passed to the exporter subprocess.

    Remove earlier report artifacts, run the selected model mode, convert run.json
    to workbook input, then export Excel. A failing child stops this sequence
    and raises to main; run.log retains its output. Success returns None after
    checking both reports are nonempty. Requests and file approvals are scripted;
    live quote clarification uses the terminal for model-generated questions.
    Its dialogue is captured in the report; run.log records the export stages.
    The file lesson modifies its own batch output file.
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
        "context.json",
        "turns-preview.png",
        "content-preview.png",
        "tree-preview.png",
        "reference-preview.png",
    ):
        (directory / filename).unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "agent_runtime",
        "--sample",
        name,
        "--out",
        ".",
        "--prices",
        str(prices),
        "--client",
        "console" if name == "quote_request" and not simulated else "static",
    ]
    if not simulated:
        command.append("--live")
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
            "reporting.excel_data",
            "run.json",
            "--out",
            "excel-data.json",
        ],
        [
            node,
            str(ROOT / "src/reporting/export_excel.mjs"),
            "excel-data.json",
            "report.xlsx",
        ],
    ]
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        for stage, command in enumerate(commands):
            # Live clarification must show the actual model question and accept
            # an answer. Redirecting its stdin to DEVNULL would silently cancel.
            interactive = stage == 0 and name == "quote_request" and not simulated
            if interactive:
                log.write(
                    "Live quote dialogue shown in terminal; see report for recorded answers.\n"
                )
                log.flush()
            subprocess.run(
                command,
                cwd=directory,
                env=env,
                stdin=None if interactive else subprocess.DEVNULL,
                stdout=None if interactive else log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    # Successful exit alone is insufficient if an exporter stopped producing files.
    for filename in ("report.html", "report.xlsx"):
        if not (directory / filename).stat().st_size:
            raise RuntimeError(f"Empty output: {filename}")


def write_index(output, results, *, simulated=False):
    """Give the batch user a navigable view of completed lessons and failures.

    main calls this initially and after each lesson so the index reflects
    progress. output is the batch directory; results contains (sample_name,
    succeeded) pairs from this run. Replace index.html with links to successful
    reports and every lesson's log; filesystem errors propagate to the caller.
    """
    mode = (
        "Simulated models and scripted human answers; no paid model calls."
        if simulated
        else "Real provider models; scripted requests and file approvals. Quote clarification uses terminal answers."
    )
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
        f"<h1>Local sample reports</h1><p>{mode} RAG uses real local retrieval. "
        "Langfuse samples are excluded. Each sample has its own standalone HTML and Excel report.</p>"
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
        default=Path("reports"),
        help="Batch directory, refreshed on rerun (default: ./reports)",
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
        help="Use a saved USD/EUR reference without refreshing it",
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="Use offline scripted models instead of real provider calls",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env.local",
        help="Shared model credentials/settings (default: repository .env.local; shell values win)",
    )
    args = parser.parse_args()
    try:
        node, env = excel_runtime()
        # Children change working directory, so resolve shared configuration here.
        # Preserve shell precedence without printing credentials or mutating .env.
        if args.env_file.exists():
            env = {
                **{
                    k: v
                    for k, v in dotenv_values(args.env_file).items()
                    if v is not None
                },
                **env,
            }
        elif args.env_file != ROOT / ".env.local":
            raise RuntimeError(f"Environment file not found: {args.env_file}")
        if not args.simulated:
            provider = env.get("LG_PROVIDER", "openai").lower()
            key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
                provider
            )
            if key is None or not env.get(key, "").strip():
                raise RuntimeError(
                    "Configure LG_PROVIDER and its API key in .env.local or the shell, or use --simulated."
                )
        prices = args.prices.resolve(strict=True)
        supplied_fx = args.fx_file or (
            Path(env["LG_FX_FILE"]) if env.get("LG_FX_FILE") else None
        )
        fx_file = (
            supplied_fx.resolve(strict=True)
            if supplied_fx
            else ROOT / "exchange-rate.json"
        )
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    output = args.out.resolve()
    # Refresh once at the network boundary, before launching any sample. Offline
    # batches and explicit rate files reuse saved data. A failed refresh cannot
    # prevent models from running or trigger another lookup in a child process.
    if not supplied_fx and not args.simulated:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/update_exchange_rate.py"),
                "--out",
                str(fx_file),
            ],
            env=env,
            check=False,
        )
    output.mkdir(parents=True, exist_ok=True)
    results = []
    # Replace a previous batch's index immediately, so unfinished lessons are
    # not advertised as successes while this batch is still running.
    write_index(output, results, simulated=args.simulated)
    print(
        f"Model mode: {'simulated' if args.simulated else 'real provider'}", flush=True
    )
    print(f"Batch output: {output}", flush=True)
    for name in SAMPLES:
        print(f"Running {name} ...", flush=True)
        try:
            run_sample(
                name,
                output / name,
                prices=prices,
                fx_file=fx_file,
                node=node,
                env=env,
                simulated=args.simulated,
            )
            ok = True
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            ok = False
            print(
                f"  FAILED ({type(exc).__name__}); see {output / name / 'run.log'}",
                flush=True,
            )
        results.append((name, ok))
        write_index(output, results, simulated=args.simulated)
        if ok:
            print(f"  HTML + Excel: {output / name}", flush=True)
    print(
        f"{sum(ok for _, ok in results)}/{len(SAMPLES)} samples completed. Open {output / 'index.html'}"
    )
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
