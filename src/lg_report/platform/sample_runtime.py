"""Shared command-line and recording plumbing, not sample application logic.

Applications compose shared agents and tools with their own test fixtures. Students can
replace the reporting wrapper without rewriting the agent. Sharing this wrapper
also keeps all samples consistent about configuration, prices, and saved evidence.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from lg_report.platform.console_client import ConsoleClient
from lg_report.platform.conversation import Conversation
from lg_report.report.exchange import get_exchange_rate
from lg_report.report.price_refresh import get_prices
from lg_report.report.pricing import Prices
from lg_report.report.recording import record_run


@dataclass(frozen=True)
class Settings:
    """Carry one run's configuration and price snapshot into the reporting wrapper.

    ``live=False`` selects scripted responses; ``capture_content=False`` omits
    captured request/response bodies, not token usage. ``output`` is the working
    directory by default;
    ``overwrite`` permits replacing its four report files. Explicit output
    directories must be new. For example, Settings(False, out, prices,
    True) records a readable offline teaching run. Freezing the container avoids
    accidental reassignment; the Prices object itself remains mutable. ``output``
    is the report root and must be owned by the caller run; this value does not
    itself create directories or files.
    """

    live: bool
    output: Path
    prices: Prices
    capture_content: bool
    overwrite: bool = False


def argument_parser(app_file: str, description: str) -> argparse.ArgumentParser:
    """Build shared CLI options so a sample can add client-specific choices.

    Parsing is left to the caller; app_file anchors the default .env location.
    This performs no network, file reads, or model initialization.
    """
    app_dir = Path(app_file).resolve().parent
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--live", action="store_true", help="Use the provider configured in .env"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="New report directory; default: working directory (replaces report files)",
    )
    parser.add_argument("--env-file", type=Path, default=app_dir / ".env")
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--fx-file", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    return parser


def settings_for(app_file: str, description: str, *, args=None) -> Settings:
    """Prepare a sample run with explicit configuration and usable cost references.

    Pass the sample's ``__file__`` and its argparse description. Optional args
    must come from argument_parser (possibly extended by the sample); omitting
    args parses the standard CLI here. Reference/configuration paths are anchored
    to the sample package so launching from another working directory still finds
    its .env and the project catalog. Run outputs default to the working directory. Shell values take precedence over .env.
    Price/FX lookups can access the network and write daily caches unless explicit
    files are supplied. FX failure leaves USD accounting available and records why
    EUR cannot be calculated. Invalid pricing files propagate their load errors;
    argparse owns invalid CLI arguments. This does not invoke an LLM. Only the
    documented FX availability failure is retained as an accounting diagnostic
    so USD reporting can continue.
    """
    app_dir = Path(app_file).resolve().parent
    project_root = app_dir.parents[1]
    # Standalone samples can extend the common parser before resolving settings;
    # existing entry points still parse their own command line here.
    if args is None:
        args = argument_parser(app_file, description).parse_args()
    # The sample's own .env is its configuration boundary. Shell variables win.
    load_dotenv(args.env_file, override=False)
    # A CLI file wins over LG_PRICES. Only a nonempty environment value becomes
    # a Path: absent/empty means automatic refresh, not the current directory.
    prices = get_prices(
        project_root / "models.json",
        supplied_file=args.prices
        or (Path(os.environ["LG_PRICES"]) if os.getenv("LG_PRICES") else None),
        cache_dir=Path(
            os.getenv("LG_PRICES_CACHE", str(project_root / ".cache/lg-report/prices"))
        ),
    )
    for model_id, refresh_error in prices.refresh_errors.items():
        print(f"{model_id}: {refresh_error}")
    try:
        # The same precedence applies to FX. None tells the FX loader to use its
        # daily cache/service; a supplied path deliberately keeps it offline.
        exchange_rate_file = args.fx_file or (
            Path(os.environ["LG_FX_FILE"]) if os.getenv("LG_FX_FILE") else None
        )
        prices.exchange = get_exchange_rate(
            exchange_rate_file,
            Path(os.getenv("LG_FX_CACHE", str(project_root / ".cache/lg-report/fx"))),
        )
    except (OSError, ValueError, KeyError) as exc:
        # USD accounting remains useful when the public daily FX service is unavailable.
        prices.exchange_error = (
            f"Daily EUR conversion unavailable ({type(exc).__name__})"
        )
    # Defaults are deliberately visible where the command was launched. A named
    # --out directory retains the archival behavior and must not already exist.
    output = args.out or Path.cwd()
    print(f"Report directory: {output.resolve()}")
    if args.out is None:
        print(
            "Replaces report.html, run.json, spans.jsonl, and prices.json in this directory."
        )
    return Settings(args.live, output, prices, not args.metadata_only, args.out is None)


def launch_local(*, app_file, description, create_run, make_static_client, title):
    """Run a sample conversation and save the evidence needed to inspect its cost.

    Sample entry points call this after defining their graph/client factories.
    It returns no result; the useful outputs are report files and client output.

    create_run(live) returns (compiled workflow, provider name, model ID).
    The workflow already contains its model adapters and tools; the two strings
    tell the recorder how to identify model pricing, not how to create agents.
    make_static_client() returns a fresh user-side client for this run.
    The static-client
    factory owns test requests; the platform never imports sample scenarios.
    Console requires a live model because fixed responses cannot answer new input.
    The function owns CLI/configuration and report finalization, while the
    supplied factories own graph construction and scripted scenario content.
    Provider, pricing, recorder, and graph failures propagate after any report
    artifacts already written by the recorder remain available for diagnosis.
    """
    parser = argument_parser(app_file, description)
    parser.add_argument("--client", choices=("static", "console"), default="static")
    args = parser.parse_args()
    # Fail before pricing or model setup for a meaningless input/model pairing.
    if args.client == "console" and not args.live:
        parser.error(
            "The console client requires --live; use static for offline examples."
        )
    settings = settings_for(app_file, description, args=args)
    graph, provider, model_id = create_run(settings.live)
    # Clients consume input differently but share identical history and recording.
    if args.client == "console":
        print("Console chat · /attach PATH (UTF-8 text), /send, /quit")
        client = ConsoleClient()
    else:
        client = make_static_client()
    try:
        record_run(
            Conversation(graph, client),
            {},
            settings.output,
            settings.prices,
            provider=provider,
            model=model_id,
            title=title,
            demo=not settings.live,
            include_output=settings.capture_content,
            overwrite=settings.overwrite,
        )
    finally:
        # A failed run can still leave useful diagnostic artifacts; avoid dead links.
        if (settings.output / "report.html").exists():
            print((settings.output / "report.html").resolve())
