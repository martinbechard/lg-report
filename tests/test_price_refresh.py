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

from reporting import price_refresh as refresh

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
    prices = refresh.get_prices(FIXTURES / "catalog.json", cache_dir=tmp_path)
    assert not prices.refresh_errors
    assert len(calls) == 4
    assert prices.models["anthropic:claude-sonnet-5"].cache_write_5m == Decimal("2.50")
    assert all(
        r.as_of == datetime.now().astimezone().date() for r in prices.models.values()
    )
    assert all(r.fetched_at for r in prices.models.values())
    again = refresh.get_prices(FIXTURES / "catalog.json", cache_dir=tmp_path)
    assert again == prices
    assert len(calls) == 4
    supplied = refresh.get_prices(
        FIXTURES / "catalog.json",
        supplied_file=FIXTURES / "catalog.json",
        cache_dir=tmp_path,
    )
    assert supplied.models["openai:gpt-5.6-luna"].as_of == date(2026, 9, 18)
    assert len(calls) == 4


# Failed refreshes must preserve the last verified snapshot and date
# rather than presenting an unverified or zero-valued price.
def test_failure_keeps_latest_verified_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "fetch_text", source)
    prices = refresh.get_prices(FIXTURES / "catalog.json", cache_dir=tmp_path)
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
    original = refresh.load_prices(FIXTURES / "catalog.json")
    result = refresh.get_prices(FIXTURES / "catalog.json", cache_dir=tmp_path)
    assert len(result.refresh_errors) == 6
    assert {k: r.as_of for k, r in result.models.items()} == {
        k: r.as_of for k, r in original.models.items()
    }
    assert not list(tmp_path.glob("*/*.json"))


def empty_catalog(tmp_path):
    """Start with no known models so tests prove discovery rather than refresh."""
    path = tmp_path / "models.json"
    path.write_text('{"as_of":"2026-09-01","note":"Discovery test","models":{}}')
    return path


def used_models(*keys):
    """Represent observed calls, including repeated models, without invoking an API."""
    from reporting.schema import Run, Step, Usage

    return Run(
        id="discovery",
        title="Discovered model tariffs",
        status="ok",
        steps=[
            Step(
                id=str(index),
                name="model",
                kind="model",
                status="ok",
                start_ns=index + 1,
                end_ns=index + 2,
                provider=key.split(":")[0],
                model=key.split(":")[1],
                usage=Usage(input_tokens=100, output_tokens=20),
            )
            for index, key in enumerate(keys)
        ],
    )


def new_openai_page():
    """Use a synthetic unseen identity with the saved official page structure."""
    return source(refresh.SOURCES["openai:gpt-5.6-sol"]).replace(
        "gpt-5.6-sol", "gpt-new-test"
    )


def current_anthropic_page():
    """Exercise the current column order and linked name with adjacent prose."""
    return """<table><tr><th>Name</th><th>Input</th><th>Output</th>
    <th>5m writes</th><th>1h writes</th><th>Hits and refreshes</th></tr>
    <tr><td><a href="/docs/en/models/haiku-4-5/overview">Claude Haiku 4.5</a>
    <span>Descriptive text is not part of the model name</span></td>
    <td>$1 / MTok</td><td>$5 / MTok</td><td>$1.25 / MTok</td>
    <td>$2 / MTok</td><td>$0.10 / MTok<sup>1</sup></td></tr></table>"""


def anthropic_overview():
    """Supply exact official-table identity evidence for a dated response model."""
    return """<table><tr><th>Feature</th><th>
    <a href="/docs/en/models/haiku-4-5/overview">Claude Haiku 4.5</a>
    <span>Example description</span></th></tr>
    <tr><td>Claude API ID</td><td>claude-haiku-4-5-20251001</td></tr>
    <tr><td>Claude API alias</td><td>claude-haiku-4-5</td></tr></table>"""


def test_recording_discovers_saves_and_reuses_both_providers(monkeypatch, tmp_path):
    """Recording fills missing tariffs before HTML and persists reusable evidence."""
    from reporting import recording
    from reporting.pricing import load_prices, summarize

    seed = empty_catalog(tmp_path)
    cache = tmp_path / "cache"
    calls = []

    def fetch(url):
        """Count source requests while keeping the recording test offline."""
        calls.append(url)
        return (
            current_anthropic_page() if url == refresh.ANTHROPIC else new_openai_page()
        )

    monkeypatch.setattr(refresh, "fetch_text", fetch)
    prices = refresh.get_prices(seed, cache_dir=cache)
    run = used_models(
        "openai:gpt-new-test", "anthropic:claude-haiku-4-5", "openai:gpt-new-test"
    )
    # Normalization has its own capture tests. This boundary test fixes its output
    # and uses the real report writer and renderer to verify saved accounting.
    monkeypatch.setattr(recording, "normalize", lambda *args, **kwargs: run)
    (tmp_path / "spans.jsonl").write_text("captured evidence")
    recording.save_report(tmp_path, prices, title=run.title)
    saved = load_prices(tmp_path / "prices.json")
    assert set(saved.models) == {"openai:gpt-new-test", "anthropic:claude-haiku-4-5"}
    assert summarize(run, saved)["unpriced_calls"] == 0
    assert (tmp_path / "report.html").is_file()
    assert len(calls) == 2
    assert len(list(cache.glob("*/*.json"))) == 2
    assert len(list(cache.glob("*/*.source.txt"))) == 2
    assert all(
        rate.source and rate.fetched_at and rate.as_of for rate in saved.models.values()
    )
    # A fresh launch has no in-memory discoveries, so this proves disk reuse.
    again = refresh.get_prices(seed, cache_dir=cache)
    refresh.ensure_run_prices(run, again)
    assert again.models == saved.models
    assert len(calls) == 2
    assert refresh.load_prices(seed).models == {}
    assert "_lookup_cache_dir" not in saved.model_dump_json()


def test_missing_lookup_failure_is_unknown_and_retries_next_launch(
    monkeypatch, tmp_path
):
    """A missing rate cannot become zero or trigger repeated requests each turn."""
    from reporting.pricing import summarize

    seed = empty_catalog(tmp_path)
    prices = refresh.get_prices(seed, cache_dir=tmp_path / "cache")
    run = used_models("openai:gpt-new-test", "anthropic:claude-haiku-4-5")
    calls = []

    def unavailable(url):
        """Fail transport once per identity for this launch."""
        calls.append(url)
        raise OSError("offline")

    monkeypatch.setattr(refresh, "fetch_text", unavailable)
    refresh.ensure_run_prices(run, prices)
    refresh.ensure_run_prices(run, prices)
    assert len(calls) == 2
    assert prices.models == {}
    assert len(prices.refresh_errors) == 2
    assert all(
        "pricing remains unknown" in error for error in prices.refresh_errors.values()
    )
    assert summarize(run, prices)["unpriced_calls"] == 2
    assert not list((tmp_path / "cache").glob("*/*.json"))
    fresh = refresh.get_prices(seed, cache_dir=tmp_path / "cache")
    refresh.ensure_run_prices(run, fresh)
    assert len(calls) == 4


def test_explicit_and_reloaded_snapshots_never_discover(monkeypatch, tmp_path):
    """An explicit offline file and saved history stay authoritative when incomplete."""
    seed = empty_catalog(tmp_path)

    def unexpected(url):
        """Any network access violates the explicit-file contract."""
        pytest.fail(f"Unexpected fetch: {url}")

    monkeypatch.setattr(refresh, "fetch_text", unexpected)
    for prices in (
        refresh.get_prices(seed, supplied_file=seed),
        refresh.load_prices(seed),
    ):
        refresh.ensure_run_prices(used_models("openai:gpt-new-test"), prices)
        assert prices.models == {}


def test_missing_alias_target_is_retrieved_without_replacing_existing_rates(
    monkeypatch, tmp_path
):
    """Explicit aliases select the lookup identity; recorded spellings never guess."""
    seed = empty_catalog(tmp_path)
    prices = refresh.get_prices(seed, cache_dir=tmp_path / "cache")
    prices.aliases["openai:my-alias"] = "openai:gpt-new-test"
    prices.models["anthropic:claude-haiku-4-5"] = refresh.Rate(
        input="123", output="456"
    )
    calls = []

    def fetch(url):
        """Only the absent alias target should need a source request."""
        calls.append(url)
        return new_openai_page()

    monkeypatch.setattr(refresh, "fetch_text", fetch)
    refresh.ensure_run_prices(
        used_models("openai:my-alias", "anthropic:claude-haiku-4-5"), prices
    )
    assert len(calls) == 1
    assert "gpt-new-test.md" in calls[0]
    assert "openai:my-alias" not in prices.models
    assert prices.models["anthropic:claude-haiku-4-5"].input == Decimal(123)


def test_anthropic_dated_identity_is_verified_and_evidence_saved(monkeypatch, tmp_path):
    """Dated response IDs need published identity evidence as well as a price row."""
    seed = empty_catalog(tmp_path)
    cache = tmp_path / "cache"
    pages = {
        refresh.ANTHROPIC: current_anthropic_page(),
        refresh.ANTHROPIC_OVERVIEW: anthropic_overview(),
    }
    monkeypatch.setattr(refresh, "fetch_text", pages.__getitem__)
    prices = refresh.get_prices(seed, cache_dir=cache)
    key = "anthropic:claude-haiku-4-5-20251001"
    refresh.ensure_run_prices(used_models(key), prices)
    assert not prices.refresh_errors
    rate = prices.models[key]
    assert (
        rate.input,
        rate.output,
        rate.cache_read,
        rate.cache_write_5m,
        rate.cache_write_1h,
    ) == (Decimal(1), Decimal(5), Decimal("0.10"), Decimal("1.25"), Decimal(2))
    assert len(list(cache.glob("*/*.models.source.txt"))) == 1
    with pytest.raises(ValueError, match="identity"):
        refresh.parse_rate(
            "anthropic:claude-haiku-4-5-20990101",
            current_anthropic_page(),
            model_text=anthropic_overview(),
        )


@pytest.mark.parametrize(
    "key", ["anthropic:claude-opus-4-7", "anthropic:claude-haiku-4-5"]
)
def test_unlisted_anthropic_canonical_models(key):
    """Existing official table rows do not need a hardcoded model allowlist."""
    rate = refresh.parse_rate(key, source(refresh.ANTHROPIC))
    assert rate.input > 0
    assert rate.output > rate.input


def test_new_openai_write_rates_and_snapshot_identity():
    """New releases use their own write evidence and explicitly listed snapshots."""
    page = new_openai_page() + "\n## Snapshots\n\n- `gpt-new-test-2026-10-01`\n"
    page = page.replace("| Output |", "| Cache writes | $5 | 1M tokens |\n| Output |")
    assert refresh.parse_rate(
        "openai:gpt-new-test-2026-10-01", page
    ).cache_write == Decimal(5)
    with pytest.raises(ValueError, match="model page"):
        refresh.parse_rate("openai:gpt-new-test-2099-01-01", page)
    with pytest.raises(ValueError, match="Conflicting"):
        refresh.parse_rate(
            "openai:gpt-new-test", page.replace("writes | $5", "writes | $9")
        )
    with pytest.raises(ValueError, match="units"):
        refresh.parse_rate(
            "openai:gpt-new-test", page.replace("writes | $5 | 1M", "writes | $5 | 1K")
        )


@pytest.mark.parametrize(
    "page",
    [
        current_anthropic_page().replace("<th>Output</th>", "<th>Something else</th>"),
        current_anthropic_page() + current_anthropic_page(),
        current_anthropic_page().replace("/ MTok", "/ KTok"),
        current_anthropic_page().replace("Claude Haiku 4.5", "Claude Haiku 4.50"),
    ],
)
def test_current_anthropic_table_rejects_ambiguous_or_changed_contract(page):
    """Wrong columns, duplicate models, units, or near-matches must stay unknown."""
    with pytest.raises(ValueError):
        refresh.parse_rate("anthropic:claude-haiku-4-5", page)
