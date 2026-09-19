"""Verify daily tariff refresh without depending on changing provider pages.

Saved source excerpts exercise strict parsing and units. Controlled fetches test
cache reuse, authoritative file overrides, and failures that must preserve old
verification dates rather than falsely advertising fresh prices.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from lg_report.report import price_refresh as refresh

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures/pricing"


# Supply saved provider HTML so parser/cache tests can run deterministically.
# The source URL selects a fixture path; return its text as fetch_text would.
def source(url):
    # OpenAI publishes one page per model, so its last URL segment names the
    # fixture. Anthropic uses one shared pricing page for all models; its exact
    # source URL selects that shared fixture instead of a model-specific file.
    return (
        FIXTURES
        / (url.rsplit("/", 1)[-1] if url != refresh.ANTHROPIC else "anthropic.html")
    ).read_text()


# Cover precedence among today's cache, fetched source, and an
# explicit authoritative file without contacting changing provider pages.
def test_daily_refresh_cache_and_override(monkeypatch, tmp_path):
    calls = []

    def fetch(url):
        # Count cache misses while supplying saved provider HTML.
        # `url` selects the fixture; no HTTP request is made by this replacement.
        calls.append(url)
        return source(url)

    monkeypatch.setattr(refresh, "fetch_text", fetch)
    prices = refresh.get_prices(ROOT / "models.json", cache_dir=tmp_path)
    assert not prices.refresh_errors
    assert len(calls) == 4
    assert prices.models["anthropic:claude-sonnet-5"].cache_write_5m == Decimal("2.50")
    assert all(
        r.as_of == datetime.now().astimezone().date() for r in prices.models.values()
    )
    assert all(r.fetched_at for r in prices.models.values())
    again = refresh.get_prices(ROOT / "models.json", cache_dir=tmp_path)
    assert again == prices
    assert len(calls) == 4
    supplied = refresh.get_prices(
        ROOT / "models.json", supplied_file=ROOT / "models.json", cache_dir=tmp_path
    )
    assert supplied.models["openai:gpt-5.6-luna"].as_of == date(2026, 9, 18)
    assert len(calls) == 4


# Failed refreshes must preserve the last verified snapshot and date
# rather than presenting an unverified or zero-valued price.
def test_failure_keeps_latest_verified_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "fetch_text", source)
    prices = refresh.get_prices(ROOT / "models.json", cache_dir=tmp_path)
    seed = tmp_path / "seed.json"
    for rate in prices.models.values():
        rate.as_of = date(2000, 1, 1)
    seed.write_text(prices.model_dump_json())
    yesterday = datetime.now().astimezone().date() - timedelta(days=1)
    for path in tmp_path.glob("*/*.json"):
        rate = refresh.Rate.model_validate_json(path.read_text())
        rate.as_of = yesterday
        rate.input = Decimal("9.125")
        path.unlink()
        path.with_name(f"{yesterday}.json").write_text(rate.model_dump_json())
    monkeypatch.setattr(refresh, "fetch_text", lambda url: "page format changed")
    result = refresh.get_prices(seed, cache_dir=tmp_path)
    assert len(result.refresh_errors) == 6
    for key in prices.models:
        assert result.models[key].as_of == yesterday
        assert result.models[key].input == Decimal("9.125")
    assert not list(tmp_path.glob(f"*/{datetime.now().astimezone().date()}.json"))


@pytest.mark.parametrize("key,url", refresh.SOURCES.items())
# A unit change invalidates arithmetic comparability, so refresh must
# reject it before replacing existing verified rates.
def test_changed_units_rejected(key, url):
    text = source(url).replace("1M tokens", "1K tokens").replace("/ MTok", "/ KTok")
    with pytest.raises(ValueError):
        refresh.parse_rate(key, text)


# New model entries and source footnotes must survive normalization,
# proving refresh can extend the ledger without dropping provenance.
def test_new_model_rates_and_footnote():
    luna = refresh.parse_rate(
        "openai:gpt-5.6-luna", source(refresh.SOURCES["openai:gpt-5.6-luna"])
    )
    sol = refresh.parse_rate(
        "openai:gpt-5.6-sol", source(refresh.SOURCES["openai:gpt-5.6-sol"])
    )
    fable = refresh.parse_rate("anthropic:claude-fable-5-1", source(refresh.ANTHROPIC))
    assert luna.cache_write == Decimal("0.25")
    assert sol.cache_write == Decimal(5)
    assert fable.cache_read == Decimal("0.25")
    assert fable.cache_write_5m == Decimal("12.50")


# Network failure is distinct from stale data; verification dates
# must remain unchanged when no fresh source can be collected.
def test_network_failure_preserves_dates(monkeypatch, tmp_path):
    def unavailable(url):
        # Force a transport failure for every requested source URL.
        # Raising rather than returning malformed text exercises the network-error path.
        raise OSError("offline")

    monkeypatch.setattr(refresh, "fetch_text", unavailable)
    original = refresh.load_prices(ROOT / "models.json")
    result = refresh.get_prices(ROOT / "models.json", cache_dir=tmp_path)
    assert len(result.refresh_errors) == 6
    assert {k: r.as_of for k, r in result.models.items()} == {
        k: r.as_of for k, r in original.models.items()
    }
    assert not list(tmp_path.glob("*/*.json"))
