"""Shared command-line and recording plumbing, not sample application logic.

Each sample owns its graph, prompts, tools, and simulator fixtures. This module
keeps environment loading and report persistence identical across those apps.
"""

import argparse
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from .exchange import get_exchange_rate
from .model_config import configured_model
from .price_refresh import get_prices
from .pricing import Prices
from .runner import ConversationAgent, record_run


@dataclass(frozen=True)
class Settings:
    """Resolved run settings: applications need no knowledge of .env precedence."""

    live: bool
    output: Path
    prices: Prices
    capture_content: bool


def settings_for(app_file: str, description: str) -> Settings:
    app_dir = Path(app_file).resolve().parent
    project = app_dir.parents[1]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--live", action="store_true", help="Use the provider configured in .env"
    )
    parser.add_argument("--out", type=Path, help="New report directory")
    parser.add_argument("--env-file", type=Path, default=app_dir / ".env")
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--fx-file", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    # The sample's own .env is its configuration boundary. Shell variables win.
    load_dotenv(args.env_file, override=False)
    prices = get_prices(
        project / "models.json",
        supplied_file=args.prices
        or (Path(os.environ["LG_PRICES"]) if os.getenv("LG_PRICES") else None),
        cache_dir=Path(
            os.getenv("LG_PRICES_CACHE", str(project / ".cache/lg-report/prices"))
        ),
    )
    for model, error in prices.refresh_errors.items():
        print(f"{model}: {error}")
    try:
        supplied = args.fx_file or (
            Path(os.environ["LG_FX_FILE"]) if os.getenv("LG_FX_FILE") else None
        )
        prices.exchange = get_exchange_rate(
            supplied,
            Path(os.getenv("LG_FX_CACHE", str(project / ".cache/lg-report/fx"))),
        )
    except (OSError, ValueError, KeyError) as exc:
        # USD accounting remains useful when the public daily FX service is unavailable.
        prices.exchange_error = (
            f"Daily EUR conversion unavailable ({type(exc).__name__})"
        )
    output = args.out or project / "reports" / app_dir.name / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    return Settings(args.live, output, prices, not args.metadata_only)


def select_model(settings: Settings, make_simulated_model):
    """Swap only the model: both modes execute the application's actual graph."""
    if settings.live:
        return configured_model()
    return make_simulated_model(), "demo", "scripted-chat"


def execute(
    graph,
    prompts: list[str],
    settings: Settings,
    *,
    provider: str,
    model: str,
    title: str,
):
    # ConversationAgent preserves messages between user turns. record_run attaches
    # callbacks at this outer boundary so nested model/tool spans share one report.
    # Keeping capture outside graph construction is the integration students can
    # reuse with an existing LangGraph application.
    try:
        record_run(
            ConversationAgent(graph, prompts),
            {},
            settings.output,
            settings.prices,
            provider=provider,
            model=model,
            title=title,
            demo=not settings.live,
            include_output=settings.capture_content,
        )
    finally:
        if (settings.output / "report.html").exists():
            print((settings.output / "report.html").resolve())
