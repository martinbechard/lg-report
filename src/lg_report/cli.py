"""Normalize saved traces and render local HTML reports."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .exchange import get_exchange_rate
from .normalize import normalize
from .pricing import load_prices
from .render import render
from .schema import Run


def main():
    parser = argparse.ArgumentParser(
        description="Normalize saved traces and render HTML reports."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--fx-file",
        type=Path,
        help="Use a USD/EUR JSON rate file without a network lookup",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render")
    p.add_argument("run", type=Path)
    p.add_argument(
        "--prices",
        type=Path,
        help="Defaults to the run's adjacent prices.json snapshot",
    )
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("normalize")
    p.add_argument("spans", type=Path)
    p.add_argument("--title", default="Imported agent run")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    try:
        exchange = None
        exchange_error = None
        if args.command != "normalize":
            try:
                rate_file = args.fx_file or (
                    Path(os.environ["LG_FX_FILE"]) if os.getenv("LG_FX_FILE") else None
                )
                exchange = get_exchange_rate(
                    rate_file, Path(os.getenv("LG_FX_CACHE", ".cache/lg-report/fx"))
                )
            except (OSError, ValueError, KeyError) as exc:
                exchange_error = (
                    f"Daily EUR conversion unavailable ({type(exc).__name__})"
                )
                print(exchange_error)
        if args.command == "normalize":
            run = normalize(args.spans, title=args.title, demo=args.demo)
            args.out.write_text(run.model_dump_json(indent=2), encoding="utf-8")
            print(args.out.resolve())
            return
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
