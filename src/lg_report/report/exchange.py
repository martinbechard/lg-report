"""Fetch and retain the latest published ECB USD/EUR daily reference rate.

A dated snapshot lets HTML and Excel explain and reproduce the same conversion.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import Field

from lg_report.report.schema import Record

# Free public ECB reference data requires no API key or account.
URL = "https://api.frankfurter.dev/v2/rate/usd/eur?providers=ecb"


class ExchangeRate(Record):
    """Keep converted report costs traceable to their exchange-rate evidence.

    rate is EUR received for one USD.

    For example, ExchangeRate(rate="0.87", date="2026-09-18") converts USD
    charges by multiplication. date identifies publication; fetched_at identifies
    retrieval. They can differ on weekends and holidays without a fetch failure.
    """

    rate: Decimal = Field(gt=0, allow_inf_nan=False)
    date: date
    fetched_at: datetime | None = None
    source: str = URL
    base: str = "USD"
    quote: str = "EUR"


def fetch_exchange_rate() -> ExchangeRate:
    """Supply report conversion with the latest published USD-to-EUR reference.

    Return a validated ExchangeRate with retrieval provenance after executing
    an HTTP request with a ten-second timeout.

    Network, decoding, and validation failures propagate. Reject wrong currency
    direction and future reference dates rather than producing plausible but
    invalid EUR estimates. Decimal parsing preserves the published precision.
    """
    request = Request(
        URL,
        headers={
            "User-Agent": "lg-report/0.1 (daily exchange-rate lookup)",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=10) as response:
        data = json.load(response, parse_float=Decimal)
    # Multiplication assumes USD → EUR. Either wrong endpoint currency would
    # invert or otherwise corrupt every converted charge, so reject the quote.
    if data.get("base") != "USD" or data.get("quote") != "EUR":
        raise ValueError("Exchange service returned an unexpected currency pair")
    result = ExchangeRate(
        rate=data["rate"], date=data["date"], fetched_at=datetime.now(UTC)
    )
    # A reference cannot have been published after retrieval; reject bad dates
    # instead of letting a future date suppress stale-data warnings.
    if result.date > result.fetched_at.date():
        raise ValueError("Exchange service returned a future reference date")
    return result


def get_exchange_rate(
    file: Path | None = None, cache_dir: Path = Path(".cache/lg-report/fx")
) -> ExchangeRate:
    """Provide a reusable conversion basis for reports started on the same day.

    Return a validated ExchangeRate from the supplied snapshot or daily cache;
    when neither exists, fetch and save a reference before returning it.

    Cache names use the machine's local calendar date, independently of the ECB's
    publication date, so weekend starts also perform at most one successful lookup.
    file is an authoritative USD-to-EUR snapshot; cache_dir holds automatic
    lookups. Read, fetch, validation, and write failures propagate to the runtime,
    which can report unavailable conversion rather than inventing a rate.
    """
    cache_path = cache_dir / f"{datetime.now().astimezone().date().isoformat()}.json"
    path = file or cache_path
    # An explicit file is authoritative even if missing (its read must fail);
    # otherwise only today's existing cache suppresses a network lookup.
    if file is not None or path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        # File provenance is useful only for user-supplied snapshots; automatic
        # cache reads must keep the provider URL recorded at fetch time.
        if file is not None:
            data.setdefault("source", str(file.resolve()))
        result = ExchangeRate.model_validate(data)
        # Supplied and cached snapshots must satisfy the same conversion
        # direction as live service responses before any costs use them.
        if result.base != "USD" or result.quote != "EUR":
            raise ValueError("Exchange-rate file must describe USD to EUR")
        return result
    result = fetch_exchange_rate()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
