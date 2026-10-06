"""Refresh standard token tariffs and discover missing models from official sources.

Parsing is deliberately strict: a changed page must not silently become a new
price. Each model retains its last verified tariff when a lookup fails.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import hashlib
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from reporting.pricing import Prices, Rate, load_prices
from reporting.schema import Run

# These bundled models remain useful source references, but are not an allowlist.
# New identities use the same strict provider page contracts below.
OPENAI_MODELS = ("gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol")
ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
ANTHROPIC_OVERVIEW = "https://platform.claude.com/docs/en/about-claude/models/overview"
COPILOT = (
    "https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing"
)
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


def source_url(key: str) -> str | None:
    """Choose an official source without letting captured IDs change the host/path.

    OpenAI dated snapshots share a model page, but parsing still requires that
    page to explicitly name the requested snapshot. Unsupported providers and
    malformed identifiers stay unpriced instead of becoming arbitrary URLs.
    """
    provider, separator, model = key.partition(":")
    if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", model):
        return None
    if provider == "openai":
        page_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
        return f"https://developers.openai.com/api/docs/models/{page_model}.md"
    if provider == "anthropic":
        return ANTHROPIC
    if provider == "copilot":
        return COPILOT
    return None


def anthropic_name(model: str, model_text: str | None = None) -> str:
    """Translate exact canonical Claude IDs into the provider's table spelling.

    Both version-first legacy names and family-first names are supported. Dated
    and latest aliases require an exact ID in the official model overview; do
    not guess which canonical tariff an unrecognized identifier represents.
    """
    if re.search(r"-(?:\d{8}|latest)$", model):
        parser = TableRows()
        parser.feed(model_text or "")
        names = []
        header = []
        for row in parser.rows:
            if not row:
                header = []
            elif row[0] == "Feature":
                header = row
            elif row and row[0] in {"Claude API ID", "Claude API alias"}:
                names.extend(
                    header[index]
                    for index, value in enumerate(row)
                    if value == model and index < len(header)
                )
        if len(set(names)) != 1:
            raise ValueError("Anthropic snapshot identity was not verified")
        return names[0]
    match = re.fullmatch(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", model)
    if match:
        family, major, minor = match.groups()
    else:
        match = re.fullmatch(r"claude-(\d+)(?:-(\d+))?-([a-z]+)", model)
        if not match:
            raise ValueError("Unrecognized Anthropic model identity")
        major, minor, family = match.groups()
    version = major + (f".{minor}" if minor else "")
    return f"Claude {family.title()} {version}"


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
        self.model_link = None
        self.in_model_link = False

    def handle_starttag(self, tag, attrs):
        """Keep price columns separate as HTMLParser encounters opening tags.

        ``tag`` is the element name; ``attrs`` identifies official model links.
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
            self.model_link = None
        elif (
            tag == "a"
            and self.cell is not None
            and dict(attrs).get("href", "").startswith("/docs/en/models/")
        ):
            # Current model cells include marketing text after the model link.
            # Keep the exact linked name rather than prefix-matching that prose.
            self.model_link = ""
            self.in_model_link = True

    def handle_data(self, data):
        """Collect tariff text for the current cell without importing page prose.

        HTMLParser supplies decoded text chunks in ``data``; append eligible
        chunks to the active cell so the closing-tag handler can retain it.
        """
        # Accept text only inside an active cell and outside a footnote; page
        # prose or superscripts could otherwise masquerade as a price.
        if self.cell is not None and not self.in_sup:
            self.cell += data
            if self.in_model_link:
                self.model_link += data

    def handle_endtag(self, tag):
        """Make completed tariff rows available for later price validation.

        HTMLParser supplies the closing element name as ``tag``. Commit a cell
        or row, or end footnote suppression, then return to the parser.
        """
        # Footnote digits are not tariff digits; row/cell boundaries reset the
        # accumulator so only one table cell contributes to each price field.
        if tag == "a":
            self.in_model_link = False
        elif tag == "sup":
            self.in_sup = False
        # Only an actually opened cell may be committed; unmatched closing tags
        # must not inject empty/misaligned columns into the extracted table.
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append((self.model_link or self.cell).strip())
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
        elif tag == "table":
            # A verified header cannot grant meaning to a later unrelated table.
            self.rows.append([])


def parse_rate(key: str, text: str, *, model_text: str | None = None) -> Rate:
    """Establish a trustworthy tariff before the refresh loop replaces saved prices.

    ``key`` is an exact provider:model identity and ``text`` is its fetched
    page body; model_text supplies official Anthropic snapshot identity evidence
    when needed. Return a Rate only when the expected pricing structure matches.

    Require exact identity, units, and a unique row so page redesigns fail closed.
    The returned Rate has amounts only: get_prices adds provenance after parsing
    succeeds. Unsupported/ambiguous content raises ValueError; absent expected
    section delimiters can raise IndexError. This function performs no network I/O.
    """
    # Validate a provider page contract, not a fixed list of model releases.
    if source_url(key) is None:
        raise ValueError("No official-source parser for this model")
    if key.startswith("copilot:"):
        return parse_copilot_rate(key.split(":", 1)[1], text)
    if key.startswith("openai:"):
        model = key.split(":", 1)[1]
        # A redirected or wrong-model page must not acquire this model's identity.
        canonical = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
        if f"Model ID: `{canonical}`" not in text or (
            model != canonical
            and f"- Default snapshot: `{model}`" not in text
            and not re.search(
                r"(?m)^- `" + re.escape(model) + r"`$", text.split("## Snapshots\n")[-1]
            )
        ):
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
        # New releases can publish a write row, a multiplier, or neither. Never
        # copy another model's multiplier; absent evidence leaves writes unknown.
        writes = re.findall(
            r"\| Cache writes \| \$(\d+(?:\.\d+)?) \| 1M tokens \|", section
        )
        multipliers = re.findall(
            r"Cache writes are billed at (\d+(?:\.\d+)?)x the uncached input token rate",
            section,
        )
        if len(writes) > 1 or len(multipliers) > 1:
            raise ValueError("Ambiguous OpenAI cache-write price")
        if "| Cache writes |" in section and not writes:
            raise ValueError("Unexpected OpenAI cache-write units")
        if writes:
            values["cache_write"] = Decimal(writes[0])
        if multipliers:
            write_rate = values["input"] * Decimal(multipliers[0])
            if writes and values["cache_write"] != write_rate:
                raise ValueError("Conflicting OpenAI cache-write prices")
            values["cache_write"] = write_rate
        if model in ("gpt-5.6-luna", "gpt-5.6-sol") and "cache_write" not in values:
            raise ValueError("Missing OpenAI cache-write price")
    elif key.startswith("anthropic:"):
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
        current_header = [
            "Name",
            "Input",
            "Output",
            "5m writes",
            "1h writes",
            "Hits and refreshes",
        ]
        layouts = {
            tuple(header): (
                "input",
                "cache_write_5m",
                "cache_write_1h",
                "cache_read",
                "output",
            ),
            tuple(current_header): (
                "input",
                "output",
                "cache_write_5m",
                "cache_write_1h",
                "cache_read",
            ),
        }
        # Select columns from the preceding verified header. The provider moved
        # output ahead of caching; relying on the old positions would misprice it.
        fields = None
        rows = []
        name = anthropic_name(key.split(":", 1)[1], model_text)
        for row in parser.rows:
            if not row:
                fields = None
            elif tuple(row) in layouts:
                fields = layouts[tuple(row)]
            elif row and row[0] == name and len(row) == 6 and fields:
                rows.append((fields, row))
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous Anthropic rate")
        values = {}
        fields, row = rows[0]
        for field, cell in zip(fields, row[1:], strict=True):
            match = re.fullmatch(r"\$(\d+(?:\.\d+)?) / MTok", cell)
            # Accept only USD per million tokens; silently reading another unit
            # would multiply every estimate by the wrong scale.
            if match is None:
                raise ValueError("Unexpected Anthropic price units")
            values[field] = Decimal(match[1])
    else:
        raise ValueError("No official-source parser for this model")
    return Rate(**values)


def parse_copilot_rate(model: str, text: str) -> Rate:
    """Read GitHub's USD token tariffs, keeping Copilot billing separate by key.

    GitHub's model names differ only in case/spaces from supported CLI codes
    (for example Claude Opus 5.5 -> claude-opus-5.5). Never erase punctuation or
    mode qualifiers to force a match. Default and long-context rows must agree
    on their boundary; unknown layouts, currencies, or duplicate rows fail closed.
    These are usage estimates before account allowances, not subscription bills.
    """
    parser = TableRows()
    parser.feed(text)
    # Validate the page-wide unit independently of bare dollar-valued cells.
    if not re.search(
        r"All prices are (?:<strong>)?per 1 million tokens(?:</strong>)?\.", text
    ):
        raise ValueError("Unexpected Copilot price units")
    prefix = ["Model", "Release status", "Category"]
    suffix = ["Input", "Cached input"]
    layouts = [
        prefix + tier + suffix + write + ["Output"]
        for tier in ([], ["Tier", "Threshold (input tokens)"])
        for write in ([], ["Cache write"])
    ]
    header = None
    rows = {}
    for row in parser.rows:
        if not row:
            header = None
        elif row in layouts:
            header = row
        elif header and row[0].lower().replace(" ", "-") == model:
            if len(row) != len(header):
                raise ValueError("Copilot pricing columns changed")
            entry = dict(zip(header, row, strict=True))
            tier = entry.get("Tier", "Default")
            if tier not in {"Default", "Long context"} or tier in rows:
                raise ValueError("Ambiguous Copilot pricing tier")
            values = {}
            for label, field in [
                ("Input", "input"),
                ("Cached input", "cache_read"),
                ("Cache write", "cache_write"),
                ("Output", "output"),
            ]:
                cell = entry.get(label)
                if label == "Cache write" and cell in {None, "Not applicable"}:
                    continue
                match = re.fullmatch(r"\$(\d+(?:\.\d+)?)", cell or "")
                if match is None:
                    raise ValueError("Unexpected Copilot price amount")
                values[field] = Decimal(match[1])
            rows[tier] = (
                values,
                entry.get("Threshold (input tokens)", "Not applicable"),
            )
    if "Default" not in rows:
        raise ValueError("Missing Copilot model rate")
    values, threshold = rows["Default"]
    if threshold != "Not applicable":
        match = re.fullmatch(r"≤ (\d+)K", threshold)
        long_values, long_threshold = rows.get("Long context", ({}, ""))
        if match is None or long_threshold != f"> {match[1]}K":
            raise ValueError("Missing or inconsistent Copilot long-context tier")
        values["long_context"] = {**long_values, "threshold": int(match[1]) * 1000}
    elif "Long context" in rows:
        raise ValueError("Unexpected Copilot long-context tier")
    return Rate(**values)


def fetch_text(url: str) -> str:
    """Supply the refresh parser with the source evidence for a tariff lookup.

    ``url`` comes from the validated provider source resolver. Execute a request with a
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
    may retry on the next launch and never advance a verification date. The
    returned snapshot also enables missing-model discovery when traces are saved.

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
    prices._lookup_cache_dir = cache_dir
    _refresh_models(prices, prices.models, cache_dir)
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


def ensure_run_prices(run: Run, prices: Prices) -> None:
    """Resolve estimated model costs before saving a run's pricing snapshot.

    This also catches models used by nested agents or resolved by a provider.
    Existing rates and explicit aliases remain authoritative. Failed identities
    are tried once per Prices lifetime so browser turns do not repeatedly block
    on an unavailable page; a fresh launch can retry. Loaded/supplied snapshots
    have no lookup cache configured and remain strictly offline. Codex usage is
    estimated at the same model's OpenAI API tariff, not at subscription charges;
    save that exact mapping as an alias so HTML and Excel share the same basis.
    """
    for step in run.steps:
        if step.kind == "model" and step.provider == "codex" and step.model:
            key = f"codex:{step.model}"
            # User-supplied Codex tariffs/aliases win. Only the provider prefix
            # changes: unknown IDs must not borrow a similarly named tariff.
            if key not in prices.models and key not in prices.aliases:
                basis = f"openai:{step.model}"
                prices.aliases[key] = prices.aliases.get(basis, basis)
    if prices._lookup_cache_dir is None:
        return
    keys = {
        prices.aliases.get(
            f"{step.provider}:{step.model}", f"{step.provider}:{step.model}"
        )
        for step in run.steps
        if step.kind == "model"
        and step.provider in {"openai", "anthropic", "copilot", "codex"}
        and step.model
    }
    missing = keys - prices.models.keys() - prices.refresh_errors.keys()
    _refresh_models(prices, missing, prices._lookup_cache_dir)


def _refresh_models(prices: Prices, keys: Iterable[str], cache_dir: Path) -> None:
    """Merge verified daily cache/fetch results into a mutable automatic snapshot.

    Missing models have no fallback: failed retrieval must leave their rate absent.
    Source bodies and parsed rates are saved together for reuse on later launches.
    Existing models keep their most recent verified fallback if refresh fails.
    """
    today = datetime.now().astimezone().date()
    pages = {}
    for key in sorted(set(keys)):
        old_rate = prices.models.get(key)
        # Derived demo entries have no independent provider page; update them
        # after their real-model basis has completed its refresh attempt.
        if old_rate and old_rate.based_on:
            continue
        url = source_url(key)
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
                    and (
                        old_rate is None
                        or cached.as_of >= (old_rate.as_of or prices.as_of)
                    )
                ):
                    prices.models[key] = cached
            # Skip fetching only when today's file exists AND the selected tariff
            # is verified today from this source; file existence alone is not proof.
            if (
                path.exists()
                and key in prices.models
                and prices.models[key].as_of == today
                and prices.models[key].source == url
            ):
                continue
            # Several models share one provider table; fetch it only once per
            # refresh pass while validating and dating each model separately.
            if url not in pages:
                pages[url] = fetch_text(url)
            body = pages[url]
            model_text = None
            if key.startswith("anthropic:") and re.search(r"-(?:\d{8}|latest)$", key):
                # A dated API response must be tied to the provider's published
                # model name before using that name's tariff. Retain both pages.
                if ANTHROPIC_OVERVIEW not in pages:
                    pages[ANTHROPIC_OVERVIEW] = fetch_text(ANTHROPIC_OVERVIEW)
                model_text = pages[ANTHROPIC_OVERVIEW]
            rate = parse_rate(key, body, model_text=model_text)
            rate.as_of = today
            rate.source = url
            rate.fetched_at = datetime.now(UTC)
            model_dir.mkdir(parents=True, exist_ok=True)
            # Retain the exact source alongside the parsed tariff for auditing.
            path.with_suffix(".source.txt").write_text(body, encoding="utf-8")
            if model_text is not None:
                path.with_suffix(".models.source.txt").write_text(
                    model_text, encoding="utf-8"
                )
            # Publish a complete parsed snapshot atomically so interruption cannot
            # leave a partial JSON file that looks like today's successful cache.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(rate.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)
            prices.models[key] = rate
            prices.refresh_errors.pop(key, None)
        except (OSError, ValueError, IndexError) as exc:
            prices.refresh_errors[key] = f"Refresh failed ({type(exc).__name__}); " + (
                "retained last verified prices."
                if key in prices.models
                else "pricing remains unknown."
            )
