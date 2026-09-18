"""Refresh supported standard token tariffs once per day from official sources.

Parsing is deliberately strict: a changed page must not silently become a new
price. Each model retains its last verified tariff when a lookup fails.
"""

import hashlib
import re
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from .pricing import Prices, Rate, load_prices

OPENAI_MODELS = ("gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol")
ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
ANTHROPIC_MODELS = {
    "anthropic:claude-sonnet-5": "Claude Sonnet 5",
    "anthropic:claude-opus-4-8": "Claude Opus 4.8",
    "anthropic:claude-fable-5-1": "Claude Fable 5.1",
}
SOURCES = {
    **{
        f"openai:{model}": f"https://developers.openai.com/api/docs/models/{model}.md"
        for model in OPENAI_MODELS
    },
    **{key: ANTHROPIC for key in ANTHROPIC_MODELS},
}


class TableRows(HTMLParser):
    """Read visible table cells, excluding scripts and serialized page copies."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.cell = None
        self.in_sup = False

    def handle_starttag(self, tag, attrs):
        if tag == "sup":
            self.in_sup = True
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None and not self.in_sup:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == "sup":
            self.in_sup = False
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)


def parse_rate(key: str, text: str) -> Rate:
    if key.startswith("openai:") and key in SOURCES:
        model = key.split(":", 1)[1]
        if f"Model ID: `{model}`" not in text:
            raise ValueError("Unexpected OpenAI model page")
        section = text.split("### Text tokens\n", 1)[1].split("\n## ", 1)[0]
        values = {}
        for label, field in [
            ("Input", "input"),
            ("Cached input", "cache_read"),
            ("Output", "output"),
        ]:
            matches = re.findall(
                r"\| " + label + r" \| \$(\d+(?:\.\d+)?) \| 1M tokens \|", section
            )
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous OpenAI rate")
            values[field] = Decimal(matches[0])
        if model in ("gpt-5.6-luna", "gpt-5.6-sol"):
            multiplier = re.search(
                r"Cache writes are billed at (\d+(?:\.\d+)?)x the uncached input token rate",
                section,
            )
            if multiplier is None:
                raise ValueError("Missing OpenAI cache-write price")
            values["cache_write"] = values["input"] * Decimal(multiplier[1])
    elif key in ANTHROPIC_MODELS:
        parser = TableRows()
        parser.feed(text)
        header = [
            "Model",
            "Base input tokens",
            "5m cache writes",
            "1h cache writes",
            "Cache hits and refreshes",
            "Output tokens",
        ]
        if header not in parser.rows:
            raise ValueError("Anthropic pricing columns changed")
        rows = [
            row
            for row in parser.rows
            if row and row[0] == ANTHROPIC_MODELS[key] and len(row) == 6
        ]
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous Anthropic rate")
        values = {}
        for field, cell in zip(
            ("input", "cache_write_5m", "cache_write_1h", "cache_read", "output"),
            rows[0][1:],
            strict=True,
        ):
            match = re.fullmatch(r"\$(\d+(?:\.\d+)?) / MTok", cell)
            if match is None:
                raise ValueError("Unexpected Anthropic price units")
            values[field] = Decimal(match[1])
    else:
        raise ValueError("No official-source parser for this model")
    return Rate(**values)


def fetch_text(url: str) -> str:
    request = Request(
        url, headers={"User-Agent": "lg-report/0.1 (daily token-price lookup)"}
    )
    with urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8")


def get_prices(
    default_file: Path,
    *,
    supplied_file: Path | None = None,
    cache_dir: Path = Path(".cache/lg-report/prices"),
) -> Prices:
    """An explicit file is authoritative and never triggers network requests."""
    if supplied_file is not None:
        return load_prices(supplied_file)
    prices = load_prices(default_file)
    prices.refresh_errors = {}
    today = datetime.now().astimezone().date()
    pages = {}
    for key, old_rate in list(prices.models.items()):
        if old_rate.based_on:
            continue
        url = SOURCES.get(key)
        if url is None:
            if not key.startswith("demo:"):
                prices.refresh_errors[key] = (
                    "No official-source parser; supply a pricing file."
                )
            continue
        model_dir = cache_dir / hashlib.sha256(key.encode()).hexdigest()[:16]
        path = model_dir / f"{today}.json"
        try:
            # Reuse the newest verified snapshot even if a later daily fetch fails.
            previous = sorted(
                p for p in model_dir.glob("*.json") if p.stem <= today.isoformat()
            )
            if previous:
                cached = Rate.model_validate_json(previous[-1].read_text())
                if (
                    cached.source == url
                    and cached.as_of
                    and cached.as_of >= (old_rate.as_of or prices.as_of)
                ):
                    prices.models[key] = cached
            if (
                path.exists()
                and prices.models[key].as_of == today
                and prices.models[key].source == url
            ):
                continue
            if url not in pages:
                pages[url] = fetch_text(url)
            body = pages[url]
            rate = parse_rate(key, body)
            rate.as_of = today
            rate.source = url
            rate.fetched_at = datetime.now(UTC)
            model_dir.mkdir(parents=True, exist_ok=True)
            # Retain the exact source alongside the parsed tariff for auditing.
            path.with_suffix(".source.txt").write_text(body, encoding="utf-8")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(rate.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)
            prices.models[key] = rate
        except (OSError, ValueError, IndexError) as exc:
            prices.refresh_errors[key] = (
                f"Refresh failed ({type(exc).__name__}); retained last verified prices."
            )
    for key, rate in list(prices.models.items()):
        if rate.based_on:
            basis = prices.models[rate.based_on]
            prices.models[key] = basis.model_copy(
                update={
                    "based_on": rate.based_on,
                    "source": f"Illustrative demo rates based on {rate.based_on}; no provider billing",
                }
            )
    return prices
