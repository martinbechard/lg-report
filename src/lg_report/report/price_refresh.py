"""Refresh supported standard token tariffs once per day from official sources.

Parsing is deliberately strict: a changed page must not silently become a new
price. Each model retains its last verified tariff when a lookup fails.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import re
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from lg_report.report.pricing import Prices, Rate, load_prices

# Only known page formats are supported; an unfamiliar model needs an explicit
# tariff instead of a best-guess scrape that could silently misprice a run.
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
    """Extract table rows so embedded page-data copies cannot duplicate prices.

    Use parser = TableRows(); parser.feed(html); then inspect parser.rows.
    Superscript footnote numbers are discarded so "$0.25 / MTok" stays a price,
    rather than gaining a spurious digit from its source note.
    """

    def __init__(self):
        """Prepare an empty price-table collector for a subsequent feed(html).

        HTMLParser owns parsing and calls the handlers below synchronously
        during feed(); constructing this object does not read a page.
        """
        super().__init__()
        self.rows = []
        self.row = []
        self.cell = None
        self.in_sup = False

    def handle_starttag(self, tag, attrs):
        """Keep price columns separate as HTMLParser encounters opening tags.

        ``tag`` is the element name; ``attrs`` contains unused HTML attributes.
        Reset the active row/cell or enter footnote suppression for later text.
        """
        # Footnote digits are not tariff digits; row/cell boundaries reset the
        # accumulator so only one table cell contributes to each price field.
        if tag == "sup":
            self.in_sup = True
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        """Collect tariff text for the current cell without importing page prose.

        HTMLParser supplies decoded text chunks in ``data``; append eligible
        chunks to the active cell so the closing-tag handler can retain it.
        """
        # Accept text only inside an active cell and outside a footnote; page
        # prose or superscripts could otherwise masquerade as a price.
        if self.cell is not None and not self.in_sup:
            self.cell += data

    def handle_endtag(self, tag):
        """Make completed tariff rows available for later price validation.

        HTMLParser supplies the closing element name as ``tag``. Commit a cell
        or row, or end footnote suppression, then return to the parser.
        """
        # Footnote digits are not tariff digits; row/cell boundaries reset the
        # accumulator so only one table cell contributes to each price field.
        if tag == "sup":
            self.in_sup = False
        # Only an actually opened cell may be committed; unmatched closing tags
        # must not inject empty/misaligned columns into the extracted table.
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)


def parse_rate(key: str, text: str) -> Rate:
    """Establish a trustworthy tariff before the refresh loop replaces saved prices.

    ``key`` is an allowlisted provider:model identity and ``text`` is its fetched
    page body; return a Rate only when the expected pricing structure matches.

    Require exact identity, units, and a unique row so page redesigns fail closed.
    The returned Rate has amounts only: get_prices adds provenance after parsing
    succeeds. Unsupported/ambiguous content raises ValueError; absent expected
    section delimiters can raise IndexError. This function performs no network I/O.
    """
    # Provider identity and an allowlisted model select a known page contract;
    # Anthropic uses its table parser, while unsupported identities fail closed.
    if key.startswith("openai:") and key in SOURCES:
        model = key.split(":", 1)[1]
        # A redirected or wrong-model page must not acquire this model's identity.
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
            # Zero matches means the expected tariff is absent; multiple matches
            # are ambiguous. Neither case justifies selecting an arbitrary price.
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous OpenAI rate")
            values[field] = Decimal(matches[0])
        # These supported models publish a separate write multiplier; other
        # models retain an unknown write tariff rather than inheriting it.
        if model in ("gpt-5.6-luna", "gpt-5.6-sol"):
            multiplier = re.search(
                r"Cache writes are billed at (\d+(?:\.\d+)?)x the uncached input token rate",
                section,
            )
            # The write price is required for this model contract; without it
            # the scrape cannot count as a newly verified complete tariff.
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
        # Positional extraction is safe only while provider columns retain their
        # expected meaning; a reordered/redesigned table must fail validation.
        if header not in parser.rows:
            raise ValueError("Anthropic pricing columns changed")
        # Ignore empty/unrelated rows and require all six columns for the exact
        # model name; partial rows cannot safely map to token categories.
        rows = [
            row
            for row in parser.rows
            if row and row[0] == ANTHROPIC_MODELS[key] and len(row) == 6
        ]
        # Missing and duplicate model rows both leave tariff selection uncertain.
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous Anthropic rate")
        values = {}
        for field, cell in zip(
            ("input", "cache_write_5m", "cache_write_1h", "cache_read", "output"),
            rows[0][1:],
            strict=True,
        ):
            match = re.fullmatch(r"\$(\d+(?:\.\d+)?) / MTok", cell)
            # Accept only USD per million tokens; silently reading another unit
            # would multiply every estimate by the wrong scale.
            if match is None:
                raise ValueError("Unexpected Anthropic price units")
            values[field] = Decimal(match[1])
    else:
        raise ValueError("No official-source parser for this model")
    return Rate(**values)


def fetch_text(url: str) -> str:
    """Supply the refresh parser with the source evidence for a tariff lookup.

    ``url`` comes from the supported-source mapping. Execute a request with a
    fifteen-second timeout and return the page body for parsing and retention.

    Return decoded UTF-8 text; network and decoding errors propagate to the
    refresh loop so the previous snapshot keeps its original verification date.
    """
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
    """Resolve tariffs for sample startup, refreshing each supported model daily.

    supplied_file is authoritative and bypasses all fetching. Otherwise start
    from default_file, prefer newer cached rates, and store successful lookups
    under cache_dir by model and local calendar day. Source bodies are retained
    beside parsed JSON for audit. Same-day successes are reused; failed attempts
    may retry on the next run and never advance a verification date.

    Per-model refresh errors are attached to the returned Prices while retaining
    available tariffs. Invalid initial/supplied files still raise: they cannot be
    silently replaced. Demo tariffs inherit their declared real-model basis.
    """
    # Explicit snapshots are reproducibility inputs; startup must not replace
    # them with network data, even when their verification date is old.
    if supplied_file is not None:
        return load_prices(supplied_file)
    prices = load_prices(default_file)
    prices.refresh_errors = {}
    today = datetime.now().astimezone().date()
    pages = {}
    for key, old_rate in list(prices.models.items()):
        # Derived demo entries have no independent provider page; update them
        # after their real-model basis has completed its refresh attempt.
        if old_rate.based_on:
            continue
        url = SOURCES.get(key)
        # Unsupported models cannot be scraped safely. Retain their old tariff
        # and flag real providers; illustrative demo entries need no web source.
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
            # Future-dated cache filenames are excluded so clock mistakes cannot
            # displace the most recent snapshot valid for today.
            previous = sorted(
                p for p in model_dir.glob("*.json") if p.stem <= today.isoformat()
            )
            # With no prior snapshot, keep the bundled tariff until fetch succeeds.
            if previous:
                cached = Rate.model_validate_json(previous[-1].read_text())
                # Trust a cached tariff only from the configured official URL,
                # with a verification date at least as recent as the current one.
                if (
                    cached.source == url
                    and cached.as_of
                    and cached.as_of >= (old_rate.as_of or prices.as_of)
                ):
                    prices.models[key] = cached
            # Skip fetching only when today's file exists AND the selected tariff
            # is verified today from this source; file existence alone is not proof.
            if (
                path.exists()
                and prices.models[key].as_of == today
                and prices.models[key].source == url
            ):
                continue
            # Several models share one provider table; fetch it only once per
            # refresh pass while validating and dating each model separately.
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
            # Publish a complete parsed snapshot atomically so interruption cannot
            # leave a partial JSON file that looks like today's successful cache.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(rate.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)
            prices.models[key] = rate
        except (OSError, ValueError, IndexError) as exc:
            prices.refresh_errors[key] = (
                f"Refresh failed ({type(exc).__name__}); retained last verified prices."
            )
    for key, rate in list(prices.models.items()):
        # Derived demo rates must follow the basis actually selected above,
        # including its unchanged old date if refresh failed.
        if rate.based_on:
            basis = prices.models[rate.based_on]
            prices.models[key] = basis.model_copy(
                update={
                    "based_on": rate.based_on,
                    "source": f"Illustrative demo rates based on {rate.based_on}; no provider billing",
                }
            )
    return prices
