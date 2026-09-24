"""Resolve configuration and accounting references for a selected sample.

Settings belong to one launch. Shell values override sample dotenv values without
mutating the process environment. Price refresh may use the network; exchange
rates are read only from saved files, with missing conversion recorded explicitly.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from reporting.exchange import get_exchange_rate
from reporting.price_refresh import get_prices
from reporting.pricing import Prices

from .argument_parser import argument_parser, resolve_live_mode


@dataclass(frozen=True)
class Settings:
    """Carry one run's configuration and price snapshot into the reporting wrapper.

    ``live=False`` selects scripted responses; ``capture_content=False`` omits
    captured request/response bodies, not token usage. ``output`` defaults to reports/<sample>;
    ``overwrite`` permits replacing the generated report bundle on each run. For example, Settings(False, out, prices,
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


def settings_for(app_file: str, description: str, *, args=None) -> Settings:
    """Prepare a sample run with explicit configuration and usable cost references.

    Pass the sample's ``__file__`` and its argparse description. Optional args
    must come from argument_parser (possibly extended by the sample); omitting
    args parses the standard CLI here. Reference/configuration paths are anchored
    to the sample package so launching from another working directory still finds
    its .env and the project catalog. Run outputs default to reports/<sample> below the working directory.
    Shell values take precedence over .env.
    Model-price lookups can access the network unless explicit prices are supplied.
    FX only reads a saved local reference. Missing/invalid FX leaves USD available and records why
    EUR cannot be calculated. Invalid pricing files propagate their load errors;
    argparse owns invalid CLI arguments. This does not invoke an LLM. Only the
    documented FX availability failure is retained as an accounting diagnostic
    so USD reporting can continue.
    """
    app_dir = Path(app_file).resolve().parent
    project_root = app_dir.parents[1]
    # Direct callers may omit parsed options; the launcher supplies them after
    # catalog-specific validation in argument_parser.parse_arguments.
    if args is None:
        args = argument_parser(app_file, description).parse_args()
    # The sample's own .env is its configuration boundary. Shell variables win.
    values = {**dotenv_values(args.env_file or app_dir / ".env"), **os.environ}
    if args.live is None:
        args.live = resolve_live_mode(args, values)
    print("Model mode: live" if args.live else "Model mode: demo (scripted responses)")
    # A CLI file wins over LG_PRICES. Only a nonempty environment value becomes
    # a Path: absent/empty means automatic refresh, not the current directory.
    prices = get_prices(
        project_root / "models.json",
        supplied_file=args.prices
        or (Path(values["LG_PRICES"]) if values.get("LG_PRICES") else None),
        cache_dir=Path(
            values.get("LG_PRICES_CACHE", str(project_root / ".cache/lg-report/prices"))
        ),
    )
    for model_id, refresh_error in prices.refresh_errors.items():
        print(f"{model_id}: {refresh_error}")
    try:
        # FX is prepared before batch execution, never fetched by a sample.
        # Explicit files still override the shared checked-in reference.
        exchange_rate_file = args.fx_file or (
            Path(values["LG_FX_FILE"])
            if values.get("LG_FX_FILE")
            else project_root / "exchange-rate.json"
        )
        prices.exchange = get_exchange_rate(exchange_rate_file)
    except (OSError, ValueError, KeyError) as exc:
        # Missing or invalid saved FX must not stop model execution.
        prices.exchange_error = (
            f"Saved EUR conversion unavailable ({type(exc).__name__})"
        )
    # One stable folder per lesson makes both individual and batch runs easy to
    # find. Context modes remain separate so the comparison survives a rerun.
    name = app_dir.name
    if name in {"edit_with_patched_state", "edit_with_reloaded_state"}:
        # These public sample IDs use hyphens; their Python folders cannot.
        name = name.replace("_", "-")
    output = args.out or Path("reports") / name
    print(f"Report directory: {output.resolve()}")
    print("Replaces the previous generated report bundle in this directory.")
    return Settings(args.live, output, prices, not args.metadata_only, True)


def prepare_sample(catalog, sample_id, args):
    """Resolve shared launch inputs before choosing console, static, or web startup.

    Return settings, workflow options, and optional replacement prompts. This
    validates custom input without constructing a model or running a graph.
    Browser startup keeps these values for its per-session workflow factory.
    """
    import json
    from copy import copy

    sample = catalog.get(sample_id)
    options = dict(sample.options)
    for key in ("source", "target", "mode"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    options.update(args.options)

    settings_args = copy(args)
    settings_args.out = args.out or Path("reports") / sample_id
    settings = settings_for(
        str(sample.directory / "sample.py"), sample.description, args=settings_args
    )
    prompts = None
    if args.request is not None:
        prompts = [args.request]
        if not args.live and prompts != catalog.prompts(sample_id):
            raise ValueError("Custom requests require --live")
    if args.values is not None:
        if not args.live:
            raise ValueError("Custom requests require --live")
        values = json.loads(args.values.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("--values must contain a JSON object")
        prompts = [json.dumps(values)]
    return settings, options, prompts
