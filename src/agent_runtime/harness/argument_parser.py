"""Define launcher arguments and validate sample-specific command-line choices.

Console, scripted, and browser launches share one option vocabulary. Parsing
selects defaults from catalog metadata without constructing workflows or loading
prices. Runtime errors remain the launcher's responsibility.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
from pathlib import Path


def argument_parser(
    app_file: str | None = None,
    description: str = "Run a discovered agent sample through console, static input, or web chat.",
) -> argparse.ArgumentParser:
    """Build options without reading configuration or starting a workflow.

    Optional app_file anchors the dotenv default for direct sample callers.
    The shared launcher leaves it unset so selection determines the sample folder.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered sample IDs and descriptions",
    )
    parser.add_argument("--sample", default="simple_chat")
    parser.add_argument(
        "--client",
        choices=("console", "static", "angular"),
        help="Client to use (default comes from sample metadata)",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(app_file).resolve().parent / ".env" if app_file else None,
    )
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--fx-file", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--values", type=Path)
    parser.add_argument("--request")
    parser.add_argument(
        "--scenario", choices=("complete", "cancel"), default="complete"
    )
    parser.add_argument("--decision", choices=("approve", "reject", "cancel"))
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--public-trace", action="store_true")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Override a workflow option",
    )
    return parser


def parse_arguments(catalog, argv=None):
    """Parse overrides and reject incompatible clients before starting a run.

    Listing needs no selected sample or model configuration. Return the parser
    alongside the namespace so runtime failures use the same CLI error format.
    argv=None reads the process command line; an explicit list supports tests.
    """
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.list:
        return parser, args
    try:
        args.options = {}
        for option in args.option:
            key, value = option.split("=", 1)
            args.options[key] = json.loads(value)
        sample = catalog.get(args.sample)
        args.client = args.client or sample.default_client
        if args.client == "console" and not args.live and not sample.interaction:
            parser.error(
                "Console conversation requires --live; use --client static for scripted prompts"
            )
        if (
            args.live
            and args.client == "static"
            and sample.interaction == "clarification"
        ):
            parser.error(
                "Use --client console with --live so a human answers model questions"
            )
        if args.public_trace and (
            sample.tracing != "langfuse" or args.client == "angular"
        ):
            parser.error("--public-trace requires a terminal Langfuse sample")
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    return parser, args
