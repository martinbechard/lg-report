"""Define launcher arguments and validate sample-specific command-line choices.

Console, scripted, and browser launches share one option vocabulary. Parsing
selects defaults from catalog metadata and provider configuration without
constructing workflows or loading prices. Runtime errors remain the launcher's responsibility.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
import os
from pathlib import Path


def argument_parser(
    app_file: str | None = None,
    description: str = "Run a static or live sample, unattended with --demo or with a human client.",
) -> argparse.ArgumentParser:
    """Build options without reading configuration or starting a workflow.

    Optional app_file anchors the dotenv default for direct sample callers.
    The shared launcher leaves it unset so selection determines the sample folder.
    """
    from .model_user import DEFAULT_USER_MODEL

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered sample IDs and descriptions",
    )
    parser.add_argument("--sample", default="simple_chat")
    parser.add_argument(
        "--client",
        choices=("console", "static", "agent", "angular"),
        help="User input: fixed text (static), a model (agent), terminal (console), or browser (angular)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        default=None,
        help="Use the real model (default with a provider API key or Codex/Copilot selection)",
    )
    mode.add_argument(
        "--static",
        action="store_true",
        help="Use fixed sample text instead of real model responses, even with credentials",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run the sample by itself; combine with --live or --static to choose its model mode")
    parser.add_argument("--user-model", nargs="?", const=DEFAULT_USER_MODEL,
                        help="Model playing the user in live runs (default: gpt-6-luna)")
    parser.add_argument("--user-provider", help="Provider for the user model (default: codex)")
    parser.add_argument("--user-turns", type=int, help="Maximum user turns including the scenario seed (default: 3)")
    qa = parser.add_mutually_exclusive_group()
    qa.add_argument("--qa", action="store_true", default=None,
                    help="Evaluate live execution with one independent Sol 6 judge; static runs skip QA")
    qa.add_argument("--no-qa", action="store_false", dest="qa",
                    help="Disable QA even when LG_QA is enabled")
    parser.add_argument("--qa-model", help="Judge model (default: gpt-6-sol)")
    parser.add_argument("--qa-provider", help="Judge provider (default: codex)")
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
        values = {**catalog.configuration(args.sample, args.env_file), **os.environ}
        args.live = resolve_live_mode(args, values)
        from .model_user import DEFAULT_USER_MODEL, DEFAULT_USER_PROVIDER

        # Live runs use an independent user agent by default. Explicit fixed-text
        # and human clients never become model users because credentials exist.
        args.user_model = (
            args.user_model or values.get("LG_USER_MODEL") or (DEFAULT_USER_MODEL if args.live else None)
        )
        args.user_provider = args.user_provider or values.get("LG_USER_PROVIDER") or DEFAULT_USER_PROVIDER
        # Demo controls unattended execution, independently of fixed/live text.
        # Static also defaults to fixed user input; an explicit human client can
        # still demonstrate approvals around scripted model decisions.
        args.client = args.client or (
            "agent" if args.live else "static" if args.static or args.demo else sample.default_client
        )
        if args.demo and args.client in {"console", "angular"}:
            parser.error("--demo runs unattended; choose --client agent or --client static")
        if args.client == "agent" and not args.live:
            parser.error("--client agent requires live models; use --live")
        if args.user_turns is None:
            args.user_turns = int(values.get("LG_USER_MAX_TURNS") or "3") if args.client == "agent" else 3
        if args.client == "agent" and args.user_turns < 1:
            raise ValueError("--user-turns must be positive")
        if args.client == "console" and not args.live and not sample.interaction:
            parser.error(
                "Console conversation requires a provider API key or --live; use --static for fixed text"
            )
        if (
            args.live
            and args.client == "static"
            and sample.interaction == "clarification"
        ):
            parser.error(
                "Live clarification requires --client agent or --client console to answer model questions"
            )
        if args.public_trace and (
            sample.tracing != "langfuse" or args.client == "angular"
        ):
            parser.error("--public-trace requires a terminal Langfuse sample")
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    return parser, args


def resolve_live_mode(args, settings):
    """Select fixed text explicitly, otherwise detect API keys or a CLI provider.

    Shell values override the selected dotenv file, matching model construction.
    Detection only checks presence; invalid live credentials must still fail in
    the provider adapter and must never silently fall back to scripted answers.
    """
    from .model_config import configured_identity

    if args.static:
        return False
    if args.live is not None:
        return args.live
    values = {**settings, **os.environ}
    provider, _ = configured_identity(settings=values)
    from .model_providers import get_provider

    return get_provider(provider).policy.is_configured(values)
