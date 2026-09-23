"""Read the shared saved USD/EUR reference without network access.

A dated snapshot lets HTML and Excel explain and reproduce the same conversion.
Only scripts/update_exchange_rate.py fetches rates; samples never wait on FX HTTP.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from reporting.schema import Record

# Free public ECB reference data requires no API key or account.
URL = "https://api.frankfurter.dev/v2/rate/usd/eur?providers=ecb"
DEFAULT_RATE_FILE = Path(__file__).resolve().parents[2] / "exchange-rate.json"


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


def get_exchange_rate(file: Path | None = None) -> ExchangeRate:
    """Read a validated local reference, never fetch or repair it implicitly.

    The default is the shared repository exchange-rate.json; an explicit file
    takes precedence. Missing or invalid data raises so the caller can mark EUR
    unavailable while retaining USD accounting. Old publication dates remain
    honest provenance and do not cause blocking refreshes during model execution.
    """
    path = file if file is not None else DEFAULT_RATE_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("source", str(path.resolve()))
    result = ExchangeRate.model_validate(data)
    if result.base != "USD" or result.quote != "EUR":
        raise ValueError("Exchange-rate file must describe USD to EUR")
    return result
