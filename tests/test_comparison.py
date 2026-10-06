"""Protect multi-run comparison evidence, accounting, and offline rendering.

Tiny recordings isolate the risks introduced by composing runs: colliding span
IDs, different tariff snapshots, missing usage, and untrusted captured content.
No model or network access is needed to verify these report contracts.
AI attribution: Generated with AI assistance by Ellis Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from decimal import Decimal
from pathlib import Path

import pytest

from reporting.cli import main
from reporting.compare import (
    comparison_charts,
    comparison_entry,
    comparison_rows,
    render_comparison,
    response_text,
)
from reporting.exchange import ExchangeRate
from reporting.pricing import Prices, Rate
from reporting.schema import Run, Step, Usage


def bundle(tmp_path, name, *, tariff="2", usage=True, priced=True):
    """Save independent evidence with deliberately repeated IDs across bundles."""
    run = Run(
        id="same-run-id", title=name, status="ok",
        steps=[
            Step(id="root", name="Workflow", kind="workflow", start_ns=0,
                 end_ns=2_000_000_000, status="ok"),
            Step(id="call", parent_id="root", name="Model", kind="model",
                 start_ns=100_000_000, end_ns=1_000_000_000, status="ok",
                 provider="fixture", model=name,
                 usage=Usage(input_tokens=100, output_tokens=20, cache_read=50,
                             reasoning=10) if usage else None,
                 request=[{"role": "human", "content": "Same question"}],
                 response=[{"role": "ai", "content": "Recorded answer"}]),
        ],
    )
    prices = Prices(
        as_of="2026-10-06", note="Fixed test prices",
        models={f"fixture:{name}": Rate(input=tariff, output="10", cache_read="1")}
        if priced else {},
    )
    directory = tmp_path / name
    directory.mkdir()
    (directory / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
    (directory / "prices.json").write_text(prices.model_dump_json(), encoding="utf-8")
    return directory / "run.json", run, prices


def test_projection_reuses_disjoint_accounting_and_wall_time(tmp_path):
    """Nested workflow spans and cache/reasoning subsets must not double bill."""
    _, run, prices = bundle(tmp_path, "a")
    entry = comparison_entry(run, prices, "a/run.json")
    assert entry["summary"]["known_cost"] == Decimal("0.00035")
    assert entry["summary"]["input_tokens"] == 100
    assert entry["summary"]["output_tokens"] == 20
    assert entry["summary"]["duration_ms"] == 2000
    assert entry["summary"]["model_calls"] == 1


def test_text_and_structured_responses_have_readable_answers():
    """Common provider content formats render text without discarding raw evidence."""
    assert response_text([
        {"content": "First"},
        {"content": [{"type": "text", "text": "Second"}, {"type": "image"}]},
    ]) == "First\n\nSecond"
    assert "No text response captured" in response_text([{"tool_calls": [{"name": "tool"}]}])


def test_three_runs_keep_separate_snapshots_and_inputs(tmp_path):
    """Different tariffs and duplicate IDs remain independent, reproducible evidence."""
    paths = [bundle(tmp_path, name, tariff=rate)[0]
             for name, rate in (("a", "2"), ("b", "4"), ("c", "6"))]
    before = {p: p.read_bytes() for p in tmp_path.glob("*/*.json")}
    out = tmp_path / "comparison.html"
    render_comparison(paths, out)
    html = out.read_text()
    assert all(f"fixture:{name}" in html for name in ("a", "b", "c"))
    assert all(amount in html for amount in ("$0.000350", "$0.000450", "$0.000550"))
    assert "Unknown FX" in html
    assert str(tmp_path) not in html
    assert all(p.read_bytes() == data for p, data in before.items())


def test_unknown_partial_failed_and_simulated_runs_are_explicit(tmp_path):
    """A failed unknown call cannot appear as a free or complete measurement."""
    a, run, _ = bundle(tmp_path, "a", usage=False)
    run.demo = True
    run.status = "error"
    run.steps[1].status = "error"
    run.steps[1].error = "Provider failed"
    a.write_text(run.model_dump_json())
    b, _, _ = bundle(tmp_path, "b", priced=False)
    out = tmp_path / "comparison.html"
    render_comparison([a, b], out)
    html = out.read_text()
    assert "Unknown FX" in html
    assert "Incomplete</span>" in html
    assert "Simulated" in html and "Provider failed" in html
    assert "Known subtotal: $0.000000" in html
    assert "Missing usage: <strong>1</strong>" in html
    # Partial usage still shows the known tokens while declaring the omission.
    run.steps.append(run.steps[1].model_copy(update={
        "id": "known", "usage": Usage(input_tokens=7, output_tokens=3),
    }))
    a.write_text(run.model_dump_json())
    render_comparison([a, b], out)
    assert "Incomplete</span>" in out.read_text()
    assert comparison_entry(run, Prices(as_of="2026-10-06", note="Missing tariffs", models={}), "a/run.json")["summary"]["input_tokens"] == 7


def test_saved_fx_and_mixed_models_are_preserved(tmp_path):
    """A run with delegates lists all model identities and keeps its saved FX."""
    a, run, prices = bundle(tmp_path, "a")
    run.steps.append(run.steps[1].model_copy(update={"id": "delegate", "model": "other"}))
    a.write_text(run.model_dump_json())
    b, _, other_prices = bundle(tmp_path, "b")
    for path, table, rate in ((a, prices, "0.8"), (b, other_prices, "0.9")):
        table.exchange = ExchangeRate(rate=rate, date="2026-10-05")
        path.with_name("prices.json").write_text(table.model_dump_json())
    out = tmp_path / "comparison.html"
    render_comparison([a, b], out)
    html = out.read_text()
    assert "fixture:other" in html
    assert "1 USD = 0.8 EUR" in html and "1 USD = 0.9 EUR" in html
    assert "€0.000315" in html


def test_captured_content_is_escaped_and_missing_content_is_honest(tmp_path):
    """Recorded HTML/JS is data even in titles, messages, errors, and provenance."""
    a, run, prices = bundle(tmp_path, "a")
    payload = '<script>alert("x")</script>'
    run.title = run.output = run.steps[1].error = payload
    run.steps[1].response = [{"content": payload}]
    run.steps[1].request = []
    prices.note = payload
    a.write_text(run.model_dump_json())
    a.with_name("prices.json").write_text(prices.model_dump_json())
    b, _, _ = bundle(tmp_path, "b")
    out = tmp_path / "comparison.html"
    render_comparison([a, b], out, title=payload)
    html = out.read_text()
    assert payload not in html
    assert html.count("<script>") == 1  # Only the authored text-only tooltip script.
    assert "&lt;script&gt;" in html
    assert "Request content not captured." in html


def test_empty_run_is_not_presented_as_free_model_performance(tmp_path):
    """No calls differs from an observed zero-token model call."""
    a, run, _ = bundle(tmp_path, "a")
    run.steps = []
    a.write_text(run.model_dump_json())
    b, _, _ = bundle(tmp_path, "b")
    out = tmp_path / "comparison.html"
    render_comparison([a, b], out)
    assert "No recorded model calls" in out.read_text()
    assert "Not available" in out.read_text()


def test_invalid_input_and_output_collision_preserve_existing_files(tmp_path):
    """Validation completes before publication and never overwrites evidence."""
    a, _, _ = bundle(tmp_path, "a")
    b, _, _ = bundle(tmp_path, "b")
    out = tmp_path / "comparison.html"
    out.write_text("Previous report")
    with pytest.raises(ValueError, match="at least two"):
        render_comparison([a], out)
    for path in (a, a.with_name("prices.json")):
        before = path.read_bytes()
        with pytest.raises(ValueError, match="overwrite"):
            render_comparison([a, b], path)
        assert path.read_bytes() == before
    b.write_text("invalid JSON")
    with pytest.raises(ValueError):
        render_comparison([a, b], out)
    assert out.read_text() == "Previous report"


def test_non_usd_prices_are_rejected(tmp_path):
    """A mislabeled estimate must fail instead of treating foreign rates as USD."""
    _, run, prices = bundle(tmp_path, "a")
    prices.currency = "EUR"
    with pytest.raises(ValueError, match="USD"):
        comparison_entry(run, prices, "a/run.json")


def test_cli_compare_is_offline_and_uses_default_output(tmp_path, monkeypatch):
    """The installed CLI path must bypass shared FX discovery for comparisons."""
    a, _, _ = bundle(tmp_path, "a")
    b, _, _ = bundle(tmp_path, "b")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["lg-report", "compare", str(a), str(b)])

    def unexpected_lookup(*args, **kwargs):
        """Fail if comparison attempts to replace saved exchange evidence."""
        pytest.fail("Comparison must not look up a replacement exchange rate")

    monkeypatch.setattr("reporting.cli.get_exchange_rate", unexpected_lookup)
    main()
    assert Path("comparison.html").is_file()


def test_chart_uses_single_report_values_and_common_scales(tmp_path):
    """Comparison changes geometry, never tariffs or retained-context arithmetic."""
    from reporting.render import conversation_turns, cost_chart

    entries = []
    for name, tariff in (("cheap", "2"), ("expensive", "20")):
        _, run, prices = bundle(tmp_path, name, tariff=tariff)
        prices.exchange = ExchangeRate(rate="0.8", date="2026-10-05")
        entries.append(comparison_entry(run, prices, f"{name}/run.json"))
    comparison_charts(entries)
    assert entries[0]["chart"]["comparison_ticks"] == entries[1]["chart"]["comparison_ticks"]
    for entry in entries:
        original = cost_chart(conversation_turns(entry["run"], entry["prices"]), entry["prices"])
        bar = entry["chart"]["bars"][0]
        assert bar["cumulative"] == original["bars"][0]["cumulative"]
        assert bar["context_tokens"] == original["bars"][0]["context_tokens"]
        assert [(s["eur"], s["label"], s["color"]) for s in bar["segments"]] == [
            (s["eur"], s["label"], s["color"]) for s in original["bars"][0]["segments"]
        ]
    assert entries[0]["chart"]["bars"][0]["comparison_cost_y"] > entries[1]["chart"]["bars"][0]["comparison_cost_y"]


def test_turn_alignment_does_not_pair_missing_turn_with_next_answer(tmp_path):
    """A missing response must leave a cell empty, not shift later responses."""
    _, a, prices = bundle(tmp_path, "a")
    _, b, _ = bundle(tmp_path, "b")
    a.steps[1].context["report_turn"] = 1
    b.steps[1].context["report_turn"] = 2
    rows, by_turn = comparison_rows([comparison_entry(run, prices, "run.json") for run in (a, b)])
    assert by_turn and [row["label"] for row in rows] == ["Turn 1", "Turn 2"]
    assert not rows[0]["cells"][1]["calls"]
    assert not rows[1]["cells"][0]["calls"]
    a.steps[1].context["report_turn"] = 2
    rows, _ = comparison_rows([comparison_entry(run, prices, "run.json") for run in (a, b)])
    assert rows[0]["prompt"] == "Same question"
    b.steps[1].request[0]["content"] = "Different question"
    rows, _ = comparison_rows([comparison_entry(run, prices, "run.json") for run in (a, b)])
    assert rows[0]["prompt"] is None
    assert rows[0]["cells"][1]["prompt"] == "Different question"


def test_context_gaps_do_not_join_histories(tmp_path):
    """The compact plot must not draw a line through an unknown context point."""
    _, run, prices = bundle(tmp_path, "a")
    prices.exchange = ExchangeRate(rate="0.8", date="2026-10-05")
    known = run.steps[1]
    run.steps += [known.model_copy(update={"id": "missing", "start_ns": 2_000_000_000,
                                         "end_ns": 3_000_000_000, "usage": None}),
                  known.model_copy(update={"id": "later", "start_ns": 4_000_000_000,
                                         "end_ns": 5_000_000_000})]
    entry = comparison_entry(run, prices, "a/run.json")
    comparison_charts([entry])
    assert entry["chart"]["token_missing"]
    assert entry["chart"]["bars"][1]["comparison_context_y"] is None
    assert entry["chart"]["comparison_context_lines"] == []


def test_report_links_are_relative_and_only_link_existing_html(tmp_path):
    """Portable output links to full reports, falling back honestly to JSON."""
    a, _, _ = bundle(tmp_path, "a #1")
    b, _, _ = bundle(tmp_path, "b")
    a.with_name("report.html").write_text("Full report")
    out = tmp_path / "comparison.html"
    render_comparison([a, b], out)
    html = out.read_text()
    assert 'href="a%20%231/report.html"' in html
    assert 'href="b/report.html"' not in html
    assert 'href="b/run.json"' in html
    assert "Recorded request and tools" not in html
    assert "Structured response evidence" not in html
    assert "Responses side by side" in html


def test_models_share_one_graph_with_distinct_series(tmp_path):
    """Models share axes while grouped stacks and line styles remain identifiable."""
    from html.parser import HTMLParser

    class Graph(HTMLParser):
        """Inspect rendered SVG structure without starting a browser or model."""

        def __init__(self):
            super().__init__()
            self.charts = 0
            self.series = []
            self.colors = []
            self.context_lines = 0

        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == "svg":
                self.charts += 1
            if "data-model-series" in attrs:
                self.series.append(attrs["data-model-series"])
            if attrs.get("class") == "cumulative-cost-line":
                self.colors.append(attrs["stroke"])
            if attrs.get("class") == "context-line":
                assert attrs["stroke-dasharray"] == "6 4"
                self.context_lines += 1

    paths = []
    entries = []
    for name in ("a", "b", "c"):
        path, run, prices = bundle(tmp_path, name)
        prices.exchange = ExchangeRate(rate="0.8", date="2026-10-05")
        run.steps.append(run.steps[1].model_copy(update={
            "id": "followup", "start_ns": 2_000_000_000, "end_ns": 3_000_000_000,
        }))
        path.write_text(run.model_dump_json())
        path.with_name("prices.json").write_text(prices.model_dump_json())
        paths.append(path)
        entries.append(comparison_entry(run, prices, str(path)))
    comparison_charts(entries)
    # Every request group reserves one horizontal slot per model.
    assert len({entry["chart"]["bars"][0]["comparison_x"] for entry in entries}) == 3
    out = tmp_path / "comparison.html"
    render_comparison(paths, out)
    graph = Graph()
    graph.feed(out.read_text())
    assert graph.charts == 1
    assert graph.series == ["1", "2", "3"]
    assert len(set(graph.colors)) == 3
    assert graph.context_lines == 3
