"""Rebuild reports from saved evidence without rerunning an agent or billing an LLM.

Normalization and presentation are separate commands because run.json is also
consumed by the Excel exporter. This CLI preserves saved model tariffs by default;
starting a new sample is the path that refreshes provider prices automatically.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from reporting.exchange import get_exchange_rate
from reporting.normalize import normalize
from reporting.pricing import load_prices
from reporting.render import render
from reporting.schema import Run


# Let a command-line user turn saved evidence into a portable run or readable
# report. This entry point selects one path, writes its artifact, and exits;
# arguments come from the process rather than function parameters.
def main():
    """Process CLI arguments and write the selected or default output file.

    ``normalize`` accepts this application's OTel SDK JSONL, not arbitrary OTLP.
    ``render`` reads a normalized run and adjacent pricing snapshot (or --prices),
    and reads a saved local FX reference unless supplied by file. It prints the output path;
    malformed input, filesystem failures, and other processing errors exit nonzero.
    ValueError/OSError messages are shown for diagnosis; unexpected exception
    payloads are redacted because they can contain provider request details.
    """
    parser = argparse.ArgumentParser(
        description="Normalize saved traces and render HTML reports."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--fx-file",
        type=Path,
        help="Use a USD/EUR JSON rate file without a network lookup",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command_parser = commands.add_parser("render")
    command_parser.add_argument(
        "run",
        type=Path,
        nargs="?",
        default=Path("run.json"),
        help="Saved run (default: ./run.json)",
    )
    command_parser.add_argument(
        "--prices",
        type=Path,
        help="Defaults to the run's adjacent prices.json snapshot",
    )
    command_parser.add_argument(
        "--out", type=Path, help="Output HTML (default: report.html beside input)"
    )
    command_parser = commands.add_parser("normalize")
    command_parser.add_argument(
        "spans",
        type=Path,
        nargs="?",
        default=Path("spans.jsonl"),
        help="Saved trace (default: ./spans.jsonl)",
    )
    command_parser.add_argument("--title", default="Imported agent run")
    command_parser.add_argument("--demo", action="store_true")
    command_parser.add_argument(
        "--out", type=Path, help="Output JSON (default: run.json beside input)"
    )
    args = parser.parse_args()
    # A supplied input keeps its derived output beside the same run's evidence.
    # With no arguments both commands operate in the working directory.
    if args.out is None:
        args.out = (
            args.run.with_name("report.html")
            if args.command == "render"
            else args.spans.with_name("run.json")
        )
    load_dotenv(args.env_file, override=False)
    try:
        exchange = None
        exchange_error = None
        # Normalization only restructures trace evidence and needs no exchange
        # rate. Rendering shows EUR costs, so only that command resolves FX.
        if args.command != "normalize":
            try:
                # CLI wins over a nonempty environment setting. Neither supplied
                # means read the shared saved reference; neither path performs HTTP.
                rate_file = args.fx_file or (
                    Path(os.environ["LG_FX_FILE"]) if os.getenv("LG_FX_FILE") else None
                )
                exchange = get_exchange_rate(rate_file)
            except (OSError, ValueError, KeyError) as exc:
                exchange_error = (
                    f"Saved EUR conversion unavailable ({type(exc).__name__})"
                )
                print(exchange_error)
        # This command stops at portable run.json so other exporters can consume
        # the trace without HTML or any pricing lookup being a prerequisite.
        if args.command == "normalize":
            run = normalize(args.spans, title=args.title, demo=args.demo)
            args.out.write_text(run.model_dump_json(indent=2), encoding="utf-8")
            print(args.out.resolve())
            return
        # Rendering reuses saved model tariffs to preserve the estimate's basis.
        # Only an explicit --prices replaces them; FX was resolved separately.
        if args.command == "render":
            run = Run.model_validate_json(args.run.read_text(encoding="utf-8"))
            prices = load_prices(args.prices or args.run.with_name("prices.json"))
            prices.exchange, prices.exchange_error = exchange, exchange_error
            render(run, prices, args.out)
            print(args.out.resolve())
            return
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except Exception as exc:  # noqa: BLE001 - CLI boundary redacts provider failures
        # Avoid printing provider exceptions that can embed request content or credentials.
        parser.exit(
            1,
            f"Report processing failed ({type(exc).__name__}). Inspect the local report if available.\n",
        )
