"""Refresh the shared USD/EUR reference before a batch, never during a sample.

This is the only exchange-rate network boundary. Write a validated snapshot
atomically so a failed lookup preserves the previous checked-in reference.
Samples consume that local file and can still run with unknown EUR if it is absent.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

from reporting.exchange import DEFAULT_RATE_FILE, URL, ExchangeRate


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


def main():
    """Refresh one saved reference; failure leaves the previous file untouched.

    The batch invokes this once before its children. Return nonzero on failure
    so the caller can announce reuse of the saved rate without stopping models.
    No model calls or credentials are needed by this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_RATE_FILE)
    args = parser.parse_args()
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    try:
        rate = fetch_exchange_rate()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(rate.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    except (OSError, ValueError, KeyError) as exc:
        temporary.unlink(missing_ok=True)
        print(
            f"Exchange-rate refresh failed ({type(exc).__name__}); saved reference unchanged. Samples can continue; EUR is unknown if no valid reference exists."
        )
        return 1
    print(
        f"Saved {args.out.resolve()}: 1 USD = {rate.rate} EUR (reference {rate.date})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
