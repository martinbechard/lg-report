"""Fetch and retain the latest published ECB USD/EUR daily reference rate."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import Field

from .schema import Record

URL = "https://api.frankfurter.dev/v2/rate/usd/eur?providers=ecb"


class ExchangeRate(Record):
    rate: Decimal = Field(gt=0, allow_inf_nan=False)
    date: date
    fetched_at: datetime | None = None
    source: str = URL
    base: str = "USD"
    quote: str = "EUR"


def fetch_exchange_rate() -> ExchangeRate:
    request = Request(
        URL,
        headers={
            "User-Agent": "lg-report/0.1 (daily exchange-rate lookup)",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=10) as response:
        data = json.load(response, parse_float=Decimal)
    if data.get("base") != "USD" or data.get("quote") != "EUR":
        raise ValueError("Exchange service returned an unexpected currency pair")
    result = ExchangeRate(
        rate=data["rate"], date=data["date"], fetched_at=datetime.now(UTC)
    )
    if result.date > result.fetched_at.date():
        raise ValueError("Exchange service returned a future reference date")
    return result


def get_exchange_rate(
    file: Path | None = None, cache_dir: Path = Path(".cache/lg-report/fx")
) -> ExchangeRate:
    """A supplied file bypasses the network; otherwise reuse today's successful lookup.

    Cache names use the machine's local calendar date, independently of the ECB's
    publication date, so weekend starts also perform at most one successful lookup.
    """
    cache_path = cache_dir / f"{datetime.now().astimezone().date().isoformat()}.json"
    path = file or cache_path
    if file is not None or path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if file is not None:
            data.setdefault("source", str(file.resolve()))
        result = ExchangeRate.model_validate(data)
        if result.base != "USD" or result.quote != "EUR":
            raise ValueError("Exchange-rate file must describe USD to EUR")
        return result
    result = fetch_exchange_rate()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
