"""Verify GitHub-specific discovery and context tiers without account billing.

Use small synthetic tables with GitHub's published column contracts. Check disk
reuse and both exporters so spreadsheet formulas cannot silently revert a long
call to default rates after opening the file.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from decimal import Decimal

import openpyxl
import pytest

from reporting import price_refresh as refresh
from reporting.excel_data import workbook_data
from reporting.export_excel import export_workbook
from reporting.pricing import Prices, cost, load_prices
from reporting.render import render
from reporting.schema import Run, Step, Usage

PAGE = """<p>All prices are <strong>per 1 million tokens</strong>.</p>
<table><tr><th>Model</th><th>Release status</th><th>Category</th><th>Tier</th>
<th>Threshold (input tokens)</th><th>Input</th><th>Cached input</th><th>Cache write</th><th>Output</th></tr>
<tr><td>GPT-6 Sol</td><td>GA</td><td>Powerful</td><td>Default</td><td>≤ 272K</td>
<td>$2</td><td>$0.20</td><td>$2.50</td><td>$10</td></tr>
<tr><td>GPT-6 Sol</td><td>GA</td><td>Powerful</td><td>Long context</td><td>&gt; 272K</td>
<td>$4</td><td>$0.40</td><td>$5</td><td>$15</td></tr></table>
<table><tr><th>Model</th><th>Release status</th><th>Category</th><th>Input</th>
<th>Cached input</th><th>Cache write</th><th>Output</th></tr>
<tr><td>Claude Opus 5.5</td><td>GA</td><td>Powerful</td><td>$4</td><td>$0.20</td>
<td>$5</td><td>$20</td></tr></table>
<table><tr><th>Model</th><th>Release status</th><th>Category</th><th>Tier</th>
<th>Threshold (input tokens)</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
<tr><td>Gemini 3.8 Flash<sup>1</sup></td><td>GA</td><td>Versatile</td><td>Default</td>
<td>Not applicable</td><td>$0.75</td><td>$0.075</td><td>$3.75</td></tr></table>"""


def call(model="gpt-6-sol", input_tokens=100):
    """Supply an inclusive input count and ordinary output for shared accounting."""
    return Step(
        id="call",
        kind="model",
        name="model",
        status="ok",
        start_ns=1,
        end_ns=2,
        provider="copilot",
        model=model,
        usage=Usage(input_tokens=input_tokens, output_tokens=10),
    )


def test_copilot_discovery_persists_distinct_prices_and_reuses_cache(
    tmp_path, monkeypatch
):
    """A Copilot key uses GitHub's table and can be restored on another launch."""
    seed = tmp_path / "models.json"
    seed.write_text(
        Prices(as_of="2026-10-05", note="Test", models={}).model_dump_json()
    )
    calls = []

    def fetch(url):
        """Assert provider isolation while counting cache misses."""
        assert url == refresh.COPILOT
        calls.append(url)
        return PAGE

    monkeypatch.setattr(refresh, "fetch_text", fetch)
    run = Run(id="test", title="Copilot", status="ok", steps=[call()])
    for _ in range(2):
        prices = refresh.get_prices(seed, cache_dir=tmp_path / "cache")
        refresh.ensure_run_prices(run, prices)
        assert set(prices.models) == {"copilot:gpt-6-sol"}
        assert not prices.refresh_errors
        assert prices.models["copilot:gpt-6-sol"].source == refresh.COPILOT
    assert len(calls) == 1
    assert len(list((tmp_path / "cache").glob("*/*.source.txt"))) == 1
    saved = tmp_path / "prices.json"
    saved.write_text(prices.model_dump_json())
    assert load_prices(saved).models == prices.models


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-6-sol", "2"),
        ("claude-opus-5.5", "4"),
        ("gemini-3.8-flash", "0.75"),
    ],
)
def test_copilot_exact_model_spelling(model, expected):
    """Provider-native codes preserve dots and resolve across upstream families."""
    rate = refresh.parse_rate(f"copilot:{model}", PAGE)
    assert rate.input == Decimal(expected)


@pytest.mark.parametrize(
    "page",
    [
        PAGE.replace("1 million tokens", "1 thousand tokens"),
        PAGE.replace("≤ 272K", "≤ 200K"),
        PAGE.replace("Long context", "Unknown tier"),
        PAGE.replace("$2.50", "2.50 credits"),
        PAGE + PAGE,
        PAGE.replace("GPT-6 Sol", "GPT-6 Sol preview"),
    ],
)
def test_copilot_malformed_or_ambiguous_prices_rejected(page):
    """Wrong units, boundaries, names, or duplicate rows cannot become a tariff."""
    with pytest.raises(ValueError):
        refresh.parse_rate("copilot:gpt-6-sol", page)


@pytest.mark.parametrize("tokens,rate", [(272000, "2"), (272001, "4")])
def test_copilot_tier_boundary(tokens, rate):
    """The long tier reprices the entire call only above the inclusive threshold."""
    prices = Prices(
        as_of="2026-10-05",
        note="Test",
        models={
            "copilot:gpt-6-sol": refresh.parse_rate("copilot:gpt-6-sol", PAGE),
        },
    )
    output = Decimal(10 if tokens == 272000 else 15)
    expected = (Decimal(tokens) * Decimal(rate) + 10 * output) / 1_000_000
    assert cost(call(input_tokens=tokens), prices) == (expected, None)


def test_copilot_long_context_html_and_excel_recalculation(tmp_path):
    """Both visible references and recalculated formulas use the saved long tier."""
    import formulas

    prices = Prices(
        as_of="2026-10-05",
        note="Test",
        models={
            "copilot:gpt-6-sol": refresh.parse_rate("copilot:gpt-6-sol", PAGE),
        },
    )
    run = Run(
        id="test", title="Copilot", status="ok", steps=[call(input_tokens=300000)]
    )
    render(run, prices, tmp_path / "report.html")
    html = (tmp_path / "report.html").read_text()
    assert "Long context: above" in html
    assert "272,000 input tokens" in html
    assert "not your subscription invoice" in html
    path = export_workbook(workbook_data(run, prices), tmp_path / "copilot.xlsx")
    cached = openpyxl.load_workbook(path, data_only=True)
    expected = cost(run.steps[0], prices)[0]
    assert cached["Turns"]["F2"].value == pytest.approx(float(expected))
    cached.close()
    model = formulas.ExcelModel().loads(str(path)).finish()
    computed = model.calculate()["'[copilot.xlsx]TURNS'!F2"].value[0, 0]
    assert computed == pytest.approx(float(expected))


def test_missing_copilot_rate_stays_unknown(tmp_path, monkeypatch):
    """An unsupported model records a retrieval failure rather than a vendor alias."""
    seed = tmp_path / "models.json"
    seed.write_text(
        Prices(as_of="2026-10-05", note="Test", models={}).model_dump_json()
    )
    prices = refresh.get_prices(seed, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(refresh, "fetch_text", lambda url: PAGE)
    run = Run(id="test", title="Copilot", status="ok", steps=[call("unknown-model")])
    refresh.ensure_run_prices(run, prices)
    assert not prices.models
    assert "copilot:unknown-model" in prices.refresh_errors
    assert cost(run.steps[0], prices)[0] is None
