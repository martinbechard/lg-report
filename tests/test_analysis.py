"""Verify token accounting, context growth, and report conversation presentation.

Use deterministic graph runs and fixed arithmetic tariffs so provider price
changes cannot alter expected sums. Checks distinguish cache subsets, reasoning,
and nested spans to catch double-counting as well as misleading display order.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from agent_runtime.agents.chat_agent import build_agent as build_chat_agent
from agent_runtime.agents.investigation_agent import build_agent as build_thinking_agent
from agent_runtime.agents.reference_chat_agent import build_agent as build_tool_agent
from reporting import exchange
from reporting.exchange import ExchangeRate, get_exchange_rate
from reporting.execute_runnable import execute_runnable
from reporting.pricing import breakdown, cost, load_prices
from reporting.render import tree_rows
from reporting.schema import Run, Step, Usage
from samples.simple_chat.sample import make_simulated_model as make_chat_model
from samples.thinking_agent.sample import (
    make_simulated_model as make_thinking_model,
)
from samples.tool_chat.sample import make_simulated_model as make_tool_model


@pytest.fixture
def prices():
    # Every accounting test uses a fixed local tariff table so assertions
    # measure normalization behavior rather than live provider price drift.
    return load_prices(Path(__file__).parent / "fixtures/accounting_prices.json")


def test_saved_exchange_rate_is_offline_even_when_old(tmp_path, monkeypatch):
    """An old saved rate is usable without waiting on a network refresh."""
    import urllib.request

    path = tmp_path / "exchange-rate.json"
    path.write_text('{"rate":"0.88","date":"2000-01-01"}')
    monkeypatch.setattr(exchange, "DEFAULT_RATE_FILE", path)

    def forbidden(*args, **kwargs):
        """Any FX network attempt inside report execution violates the boundary."""
        pytest.fail("Reading a saved rate must not access the network")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert get_exchange_rate().rate == Decimal("0.88")
    assert get_exchange_rate(path).source == str(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        get_exchange_rate()


@pytest.mark.parametrize("fails", [False, True])
def test_exchange_refresh_preserves_snapshot_on_failure(tmp_path, monkeypatch, fails):
    """Only the refresh script fetches; unsuccessful refresh never erases evidence."""
    import runpy
    import sys

    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/update_exchange_rate.py"))["main"]
    path = tmp_path / "exchange-rate.json"
    previous = '{"rate":"0.88","date":"2000-01-01"}'
    path.write_text(previous)
    rate = ExchangeRate(rate="0.9", date="2026-09-17", fetched_at=datetime.now(UTC))
    calls = []

    def fetch():
        """Substitute one deterministic service response or network failure."""
        calls.append(1)
        if fails:
            raise OSError("service unavailable")
        return rate

    monkeypatch.setitem(main.__globals__, "fetch_exchange_rate", fetch)
    monkeypatch.setattr(sys, "argv", ["update_exchange_rate.py", "--out", str(path)])
    assert main() == (1 if fails else 0)
    assert calls == [1]
    if fails:
        assert path.read_text() == previous
    else:
        assert get_exchange_rate(path) == rate
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    "data",
    [
        {"rate": "-1", "date": "2026-09-17"},
        {"rate": "0.8", "date": "2026-09-17", "base": "EUR", "quote": "USD"},
    ],
)
# Malformed pricing input must fail at the boundary instead of
# producing partially trusted accounting values.
def test_bad_rate_file_rejected(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        get_exchange_rate(path)


# Provider usage fields may overlap or omit columns; these assertions
# protect normalization from double counting while preserving reconciliation.
def test_disjoint_token_columns_reconcile(prices):
    step = Step(
        id="1",
        name="LLM",
        kind="model",
        start_ns=0,
        end_ns=1,
        status="ok",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=Usage(
            input_tokens=1000,
            output_tokens=200,
            cache_read=100,
            cache_write=500,
            cache_write_5m=200,
            cache_write_1h=300,
            reasoning=80,
        ),
    )
    parts = breakdown(step, prices)
    assert [p["tokens"] for p in parts] == [400, 100, 200, 300, 0, 120, 80]
    assert sum(p["tokens"] for p in parts) == 1200
    assert sum(p["usd"] for p in parts) == cost(step, prices)[0]


# A real tool call must retain schema descriptions, parentage, and
# aggregate totals across the rendered report.
def test_annotated_tool_tree_and_parent_totals(tmp_path, prices):
    out = tmp_path / "run"
    prices.exchange = ExchangeRate(rate="0.871", date="2026-09-17")
    execute_runnable(
        build_tool_agent({"model": make_tool_model()}),
        {"messages": [("user", "Explain ReAct")]},
        out,
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
    )
    run = Run.model_validate_json((out / "run.json").read_text())
    tool = next(s for s in run.steps if s.kind == "tool")
    assert "Return the provided text" in tool.context["description"]
    models = [s for s in run.steps if s.kind == "model"]
    assert models[0].context["requested_tools"] == ["echo_tool"]
    assert models[1].context["finish_reason"] == "stop"
    assert "report_description" in models[0].context
    assert "message_count" in models[0].context
    rows = tree_rows(run, prices)
    assert len(rows) == len(run.steps)
    assert rows[0]["total"] == sum(cost(s, prices)[0] for s in models)
    assert sum(c["tokens"] for c in rows[0]["cells"]) == sum(
        s.usage.input_tokens + s.usage.output_tokens for s in models
    )
    html = (out / "report.html").read_text()
    assert "0.871" in html and "2026-09-17" in html
    assert "Expand all" in html and 'class="toggle"' in html
    assert "Input at standard rate" in html and "Reasoning" in html


# Saved dates provide provenance, not evidence of failed retrieval. Older
# references must retain correct amounts without coloring every cost as an error.
def test_reference_dates_and_explicit_units(tmp_path, prices):
    from reporting.render import render

    prices.as_of = __import__("datetime").date(2000, 1, 1)
    for rate in prices.models.values():
        rate.as_of = __import__("datetime").date(2000, 1, 1)
    prices.exchange = ExchangeRate(rate="0.8", date="2000-01-02")
    step = Step(
        id="m",
        name="model",
        kind="model",
        start_ns=0,
        end_ns=1,
        status="ok",
        provider="demo",
        model="scripted-chat",
        usage=Usage(input_tokens=100, output_tokens=10),
    )
    run = Run(id="run", title="Old references", status="ok", steps=[step])
    output = tmp_path / "dated.html"
    render(run, prices, output)
    html = output.read_text()
    assert "<p>1 USD = 0.8 EUR" in html
    assert 'class="stale"' not in html
    assert 'class="euro stale"' not in html
    assert "Prices 2000-01-01" not in html
    assert 'class="rate-date">Verified 2000-01-01' in html
    assert "FX 2000-01-02" not in html
    assert "Rate reference date <strong>2000-01-02</strong>" in html
    assert "USD / 1M tokens" in html and "EUR / 1M tokens" in html
    assert "0.000040 USD" in html and "0.000032 EUR" in html
    assert "earlier date alone does not mean retrieval failed" in html
    # Actual retrieval failures still require a visible explanation.
    prices.exchange = None
    prices.exchange_error = "Daily EUR conversion unavailable (TimeoutError)"
    prices.refresh_errors = {"demo:scripted-chat": "Price refresh failed"}
    render(run, prices, output)
    unavailable = output.read_text()
    assert (
        'class="notice">Daily EUR conversion unavailable (TimeoutError)' in unavailable
    )
    assert "EUR amounts are unknown" in unavailable
    assert "Price refresh failed" in unavailable


# Unknown child pricing must remain visible as partial knowledge while
# known subtotals still reconcile to their parent activity.
def test_partial_costs_reconcile_with_parent(prices):
    from reporting.pricing import summarize

    step = Step(
        id="m",
        name="model",
        kind="model",
        start_ns=0,
        end_ns=1,
        status="ok",
        provider="demo",
        model="scripted-chat",
        usage=Usage(input_tokens=100, output_tokens=10, cache_write=20),
    )
    run = Run(id="r", title="Partial", status="ok", steps=[step])
    row = tree_rows(run, prices)[0]
    assert row["partial"]
    assert row["total"] == summarize(run, prices)["known_cost"]


# The documented multi-turn sample protects context growth and report
# ordering over more than one user request.
def test_complete_multiturn_sample(tmp_path, prices):
    from fixtures.mock_client import MockClient

    from agent_runtime.harness.conversation import Conversation, Request
    from reporting.render import conversation_turns

    agent = Conversation(
        build_tool_agent({"model": make_tool_model()}),
        MockClient(
            [Request(p) for p in ["Explain ReAct", "Why do observations help?"]]
        ),
    )
    out = tmp_path / "conversation"
    execute_runnable(
        agent,
        {},
        out,
        prices,
        provider="demo",
        model="scripted-chat",
        include_output=True,
        demo=True,
    )
    run = Run.model_validate_json((out / "run.json").read_text())
    assert all(s.context.get("description") for s in run.steps)
    models = [s for s in run.steps if s.kind == "model"]
    assert len(models) == 4 and all(s.effort == "fast" for s in models)
    turns = conversation_turns(run, prices)
    assert [t["request"] for t in turns] == [
        "Explain ReAct",
        "Why do observations help?",
    ]
    assert [len(t["events"]) for t in turns] == [3, 3]
    from agent_runtime.harness.demo_meter import message_units

    assert models[0].usage.cache_read == 0
    for previous, current in pairwise(models):
        assert current.request[: len(previous.request)] == previous.request
        appended = current.request[len(previous.request) :]
        assert current.usage.input_tokens > previous.usage.input_tokens
        assert (
            current.usage.cache_read
            == previous.usage.input_tokens + previous.usage.output_tokens
        )
        assert current.usage.input_tokens - previous.usage.input_tokens == sum(
            len(message_units(message)) for message in appended
        )
        assert appended[0] == previous.response[0]
    assert any(m["role"] == "tool" for m in models[1].request)
    assert models[2].request[-1]["role"] == "human"
    for turn in turns:
        for index, cell in enumerate(turn["cells"]):
            assert cell["tokens"] == sum(
                e["cells"][index]["tokens"] for e in turn["events"]
            )
            assert cell["usd"] == sum(e["cells"][index]["usd"] for e in turn["events"])

    from reporting.pricing import summarize

    assert sum(t["total"] for t in turns) == summarize(run, prices)["known_cost"]
    assert all(e["step"].response for t in turns for e in t["events"])
    html = (out / "report.html").read_text()
    assert "No annotations" not in html and "Operation context" not in html
    assert "Span ID" in html and "Description" in html
    # Model identity and reported effort now belong to the operation cell.
    assert "<th>Model · effort</th>" not in html
    assert "scripted-chat · fast" in html and "Turn 2" in html
    assert "Tool arguments" in html and "Tool call: echo_tool" in html
    conversation = html.split("<h2>Conversation</h2>")[1].split(
        "<h2>Execution tree</h2>"
    )[0]
    assert "<table" not in conversation
    assert 'class="event llm-event' in conversation
    assert 'class="event tool-event' in conversation
    assert "Tool result → model input" not in conversation
    assert "Output · including tool calls" in conversation
    assert 'class="input-detail"' in conversation
    assert "Conversation history · cache read" in conversation
    for turn in turns:
        tool_event = turn["events"][1]
        model_event = turn["events"][2]
        assert (
            sum(p["tokens"] for p in model_event["input_parts"])
            == model_event["growth"]["delta"]
        )
        assert model_event["input_parts"][1]["tokens"] == tool_event["result_units"]
    assert "LLM request · " in conversation
    assert "LLM invocation" not in conversation
    assert "Initial setup" not in conversation
    assert "Added to context" not in conversation
    assert conversation.count("Context: Added:") == 8
    assert "System prompt" in conversation
    assert "Tool definitions" in conversation
    assert (
        conversation.index("Tool definitions")
        < conversation.index("System prompt")
        < conversation.index("User prompt ·")
    )
    assert conversation.count("User prompt ·") == 2
    assert "Call ID:" not in conversation
    first_call = conversation.split('class="event llm-event')[1].split(
        '<div class="status">Status:'
    )[0]
    assert first_call.index("Input token costs") < first_call.index("LLM response")
    assert (
        first_call.index("LLM response")
        < first_call.index("Output token costs")
        < first_call.index("Call cost")
    )
    assert "Output · including tool calls" not in first_call.split("LLM response")[0]
    assert "Simulated input composition" not in conversation
    assert "first observed call" not in conversation
    assert "unchanged prefix messages" not in conversation
    assert "simulated input tokens" not in conversation
    assert "Cache write · other" not in html
    assert "Prices 20" not in conversation and "FX 20" not in conversation


# Provider invocation metadata is the source of truth for effort and
# must be preserved in normalized steps and the generated report.
def test_effort_from_provider_invocation(tmp_path):
    from uuid import uuid4

    from agent_runtime.harness.trace_capture import TraceCapture
    from reporting.normalize import normalize

    path = tmp_path / "trace.jsonl"
    capture = TraceCapture(path, "openai", "test-model")
    capture.on_chat_model_start(
        {}, [], run_id=uuid4(), invocation_params={"reasoning_effort": "low"}
    )
    capture.close()
    assert normalize(path, title="effort").steps[0].effort == "low"


# Simulated context must include prior model output and tool results;
# dropping either would make later token estimates misleading.
def test_context_simulation_retains_response_and_tool_result():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from agent_runtime.harness.demo_meter import (
        ContextSimulation,
        message_record,
        message_units,
    )

    simulation = ContextSimulation()
    # LangChain message objects carry conversation data: HumanMessage is user
    # input, AIMessage carries assistant output/proposed tool calls, and
    # ToolMessage carries the execution result tied to a tool-call id.
    user = message_record(HumanMessage(content="Look it up"))
    response = message_record(
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "a"}])
    )
    tool = message_record(ToolMessage(content="Found evidence", tool_call_id="a"))
    answer = message_record(AIMessage(content="Here is the answer"))
    first, _ = simulation.record([], [user], response)
    second, _ = simulation.record([], [user, response, tool], answer)
    assert first["cache_read_tokens"] == 0
    assert first["request_cache_write_tokens"] == first["fresh_input_tokens"]
    assert first["response_cache_write_tokens"] == first["response_tokens"]
    assert second["cache_read_tokens"] == first["context_after_response_tokens"]
    assert second["fresh_input_tokens"] == len(message_units(tool))
    assert second["added_message_tokens"] == len(message_units(tool))
    assert len(simulation.context) == second["context_after_response_tokens"]
    assert len(simulation.ledger) == 2
    with pytest.raises(ValueError, match="retained"):
        simulation.record([], [user], answer)
    assert len(simulation.ledger) == 2


@pytest.mark.parametrize("tool_loop", [False, True])
# Each request needs a fresh root while its model/tool components stay
# nested under that request for turn-level attribution.
def test_every_request_nests_components_under_fresh_input(tmp_path, prices, tool_loop):
    from fixtures.mock_client import MockClient

    from agent_runtime.harness.conversation import Conversation, Request

    # tool_loop selects the two paths whose fresh-input composition differs:
    # direct chat adds user messages; the tool graph also adds observations and
    # a second request per turn. Both must obey the same report nesting rules.
    out = tmp_path / "layout"
    execute_runnable(
        Conversation(
            build_tool_agent({"model": make_tool_model()})
            if tool_loop
            else build_chat_agent({"model": make_chat_model()}),
            MockClient([Request(p) for p in ["First question", "Follow-up"]]),
        ),
        {},
        out,
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
        include_output=True,
    )
    html = (out / "report.html").read_text()
    calls = html.split('class="event llm-event')[1:]
    assert len(calls) == (4 if tool_loop else 2)
    for block in calls:
        call = block.split('<div class="status">Status:')[0]
        assert call.count('class="input-detail"') == 1
        fresh = call.index("Input at standard rate")
        detail = call.index('class="input-detail"')
        write = call.index("Cache write")
        assert fresh < detail < write
        for label in [
            "Tool definitions",
            "System prompt",
            "User prompt",
            "Tool result",
        ]:
            # A request only contains its newly appended components: later
            # requests need not repeat cached definitions or the system prompt.
            # Check placement when present rather than requiring every label.
            if label in call:
                assert detail < call.index(label) < write
        # The initial request has no retained history. Subsequent requests
        # show that cached prefix before their new, uncached content.
        if "Conversation history" in call:
            assert call.index("Conversation history") < fresh


# When only a five-minute cache rate exists, unspecified cache-write
# duration must use that rate rather than becoming unpriced by accident.
def test_unspecified_cache_write_uses_five_minute_rate(prices):
    from decimal import Decimal

    step = Step(
        id="cache",
        name="model",
        kind="model",
        provider="anthropic",
        model="claude-sonnet-4-6",
        start_ns=0,
        end_ns=1,
        status="ok",
        usage=Usage(input_tokens=1000, output_tokens=100, cache_write=1000),
    )
    parts = breakdown(step, prices)
    assert parts[4]["usd"] == Decimal("0.00375")
    assert cost(step, prices)[0] == Decimal("0.00525")
    step.usage.cache_write_1h = 1000
    assert cost(step, prices)[0] == Decimal("0.0075")


# Reasoning tokens are a billed subset of model output and must survive
# the thinking sample's normalization and summary path.
def test_thinking_sample_accounts_for_reasoning(tmp_path, prices):

    out = tmp_path / "thinking"
    execute_runnable(
        build_thinking_agent({"model": make_thinking_model()}),
        {"messages": [("user", "Investigate latency")]},
        out,
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
        include_output=True,
    )
    run = Run.model_validate_json((out / "run.json").read_text())
    models = [s for s in run.steps if s.kind == "model"]
    tools = [s for s in run.steps if s.kind == "tool"]
    assert len(models) == 7 and len(tools) == 6
    heavy = models[3]
    assert heavy.usage.reasoning == 12000
    assert heavy.context["thinking_text"]
    html = (out / "report.html").read_text()
    assert heavy.context["thinking_text"] in html
    assert 'class="thinking-preview"' in html
    # Request IDs identify the model event even when HTML attributes change.
    heavy_html = next(
        block for block in re.split(r'<details[^>]*class="event ', html)
        if heavy.context["thinking_text"] in block
    )
    assert (
        heavy_html.index("LLM response")
        < heavy_html.index("Reasoning ·")
        < heavy_html.index('class="thinking-preview"')
    )
    assert heavy_html.count("Reasoning ·") == 1
    assert sum(t.start_ns < heavy.start_ns for t in tools) == 3
    assert sum(t.start_ns > heavy.start_ns for t in tools) == 3
    assert breakdown(heavy, prices)[6]["usd"] == Decimal("0.0192")
    assert (
        models[4].usage.cache_read
        == heavy.usage.input_tokens + heavy.usage.output_tokens - heavy.usage.reasoning
    )


@pytest.mark.parametrize(
    ("provider", "model", "tokens", "expected"),
    [
        ("openai", "gpt-5.5", 525_000, 50),
        ("openai", "gpt-5.6-luna", 0, 0),
        ("openai", "gpt-5.6-sol", 1_050_000, 100),
        ("anthropic", "claude-sonnet-5", 250_000, 25),
        ("anthropic", "claude-opus-4-8", 250_000, 25),
        ("anthropic", "claude-fable-5-1", 250_000, 25),
        ("openai", "unknown", 100, None),
    ],
)
def test_context_occupancy_uses_inclusive_input(
    prices, provider, model, tokens, expected
):
    """Cached tokens count once; output and unfamiliar models cannot distort occupancy."""
    from reporting.context import context_utilization

    step = Step(
        id="m",
        name="model",
        kind="model",
        start_ns=0,
        end_ns=1,
        status="ok",
        provider=provider,
        model=model,
        usage=Usage(input_tokens=tokens, output_tokens=200, cache_read=tokens),
    )
    result = context_utilization(step, prices)
    if expected is None:
        assert result is None
    else:
        assert result["percent"] == expected
        assert result["tokens"] == tokens
    step.usage = None
    assert context_utilization(step, prices) is None


def test_context_chart_gaps_axes_and_thinner_line(tmp_path, prices):
    """Exercise real rendering with missing, zero, half-full, and overfull inputs."""
    from reporting.render import conversation_turns, cost_chart, render

    prices.exchange = ExchangeRate(rate="0.8", date="2026-09-20")
    # Reuse the arithmetic tariff while declaring its model basis explicitly;
    # the ratio must remain labelled illustrative for these simulated requests.
    prices.models["demo:scripted-chat"].based_on = "openai:gpt-5.6-luna"
    run = Run(
        id="context",
        title="Context chart",
        status="ok",
        demo=True,
        steps=[
            Step(
                id=str(i),
                name="model",
                kind="model",
                start_ns=i,
                end_ns=i + 1,
                status="ok",
                provider="demo",
                model="scripted-chat",
                usage=None
                if count is None
                else Usage(input_tokens=count, output_tokens=10),
            )
            for i, count in enumerate([525_000, None, 0, 1_575_000])
        ],
    )
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert len(chart["context_lines"]) == 3
    assert chart["context_missing"] and chart["context_illustrative"]
    assert chart["bars"][0]["context"]["percent"] == pytest.approx(525_010 / 1_050_000 * 100)
    assert chart["bars"][2]["context_y"] < 290
    assert chart["bars"][3]["context_y"] == 30
    assert chart["context_ticks"][-1]["value"] == pytest.approx(1_575_010 / 1_050_000 * 100)
    assert len(chart["token_lines"]) == 3
    assert chart["token_max"] == 1_800_000
    assert chart["bars"][2]["token_y"] < 290
    assert chart["bars"][3]["token_y"] > 30
    destination = tmp_path / "context.html"
    render(run, prices, destination)
    html = destination.read_text()
    assert 'stroke="#17734b" stroke-width="1.5"' in html
    assert 'stroke="#912c42" stroke-width="3"' in html
    assert "50.001%" in html and "0.001%" in html
    assert "Illustrative Post-call Context" in html and "Context %" in html
    assert "gaps are not zero" in html
    assert '<input type="checkbox" id="context-percent-toggle">' in html
    assert 'data-context-mode="percent" style="display:none"' in html
    assert 'data-context-mode="tokens"' in html


@pytest.mark.parametrize("count", [None, 0, 1, 1226])
def test_raw_context_chart_does_not_require_known_capacity(prices, count):
    """Raw usage stays available for unknown models; empty/zero axes stay finite."""
    from reporting.render import conversation_turns, cost_chart

    prices.exchange = ExchangeRate(rate="0.8", date="2026-09-20")
    run = Run(id="raw", title="Raw", status="ok", steps=[
        Step(id="m", name="model", kind="model", start_ns=0, end_ns=1,
             status="ok", provider="unknown", model="unknown",
             usage=None if count is None else Usage(input_tokens=count, output_tokens=0))
    ])
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert chart["context_missing"]
    assert chart["bars"][0]["context_tokens"] == count
    assert chart["token_missing"] is (count is None)
    assert chart["token_max"] > (count or 0)
    if count == 1226:
        assert chart["token_max"] == 1400


@pytest.mark.parametrize("lifetime_buckets", [False, True])
def test_cost_chart_tooltips_break_down_inclusive_token_totals(tmp_path, prices, lifetime_buckets):
    """Every chart mark shares one readable, nonduplicated token explanation."""
    from html.parser import HTMLParser

    from reporting.render import conversation_turns, cost_chart, render

    prices.exchange = ExchangeRate(rate="0.8", date="2026-09-20")
    prices.models["demo:scripted-chat"].based_on = "openai:gpt-5.6-luna"
    # Price every cache bucket so every displayed bar segment is exercised.
    rate = prices.models["demo:scripted-chat"]
    rate.cache_write = Decimal(2)
    rate.cache_write_5m = Decimal(3)
    rate.cache_write_1h = Decimal(4)
    run = Run(id="breakdown", title="Token breakdown", status="ok", steps=[
        Step(id="m", name="model", kind="model", start_ns=0, end_ns=1,
             status="ok", provider="demo", model="scripted-chat",
             usage=Usage(input_tokens=1000, cache_read=200, cache_write=100,
                         cache_write_5m=30 if lifetime_buckets else 0,
                         cache_write_1h=20 if lifetime_buckets else 0,
                         output_tokens=300, reasoning=50))
    ])
    destination = tmp_path / "breakdown.html"
    render(run, prices, destination)

    class Tooltips(HTMLParser):
        """Read rendered attributes so escaping cannot hide a broken tooltip."""
        def handle_starttag(self, tag, attrs):
            value = dict(attrs).get("data-cost-tip", "")
            if value.startswith("Request 1 ·"):
                tips.append(value)
                if tag == "rect":
                    bar_tips.append(value)

    tips = []
    bar_tips = []
    Tooltips().feed(destination.read_text())
    assert len(tips) >= 4  # Cost segments, cumulative point, and both context modes.
    for tip in tips:
        for expected in ("• Post-call Context: 1,250", "Total input: 1,000",
                         "Previous context: 0 tokens",
                         "• Fresh input: 800", "• Cache read: 200",
                         "Cache write: 100",
                         "Output: 300", "• Non-reasoning output: 250", "• Reasoning: 50"):
            assert expected in tip
        # Every category appears once, with its count on the same line. The
        # prior context is not an extra additive amount in the billing buckets.
        lines = tip.split(" | ")
        for label in ("• Fresh input", "• Cache read", "Cache write",
                      "Output", "• Non-reasoning output", "• Reasoning", "• Post-call Context"):
            matches = [line for line in lines if line.startswith(label + ":")]
            assert len(matches) == 1
            assert "tokens" in matches[0]
        assert lines.index("Output: 300 tokens") < lines.index("• Non-reasoning output: 250 tokens")
        assert lines.index("• Reasoning: 50 tokens") < lines.index("• Post-call Context: 1,250 tokens")
        assert "Token breakdown for this request" not in tip
        assert "(this segment)" not in tip
        assert "Reasoning item retained" not in tip
        assert "Output (including reasoning)" not in tip
    assert len(bar_tips) == (7 if lifetime_buckets else 5)
    assert len(set(bar_tips)) == 1
    chart = cost_chart(conversation_turns(run, prices), prices)
    bar = chart["bars"][0]
    assert bar["segments"][0]["label"] == "Fresh input"
    assert bar["segments"][0]["tokens"] == 800
    assert sum(segment["eur"] for segment in bar["segments"]) == bar["total"]
    assert sum(segment["eur"] for segment in bar["segments"][2:5]) == (
        sum(cell["usd"] for cell in breakdown(run.steps[0], prices)[2:5])
        - Decimal(100) * rate.input / Decimal(1_000_000)
    ) * prices.exchange.rate
    html = destination.read_text()
    assert ".cost-tip-detail{font-size:11px" in html
    assert "text.startsWith('• ')" in html
    # A retained reasoning item changes only the estimated carried context;
    # billing breakdown rows still report the same complete model output.
    run.steps[0].response = [{"content": [
        {"type": "reasoning", "encrypted_content": "opaque"},
        {"type": "text", "text": "answer"},
    ]}]
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert chart["bars"][0]["context_tokens"] == 1300
    assert "• Post-call Context: 1,300 tokens" in chart["bars"][0]["token_breakdown"]
    assert "• Reasoning: 50 tokens (added to context)" in chart["bars"][0]["token_breakdown"]
    run.steps[0].usage = None
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert chart["bars"][0]["token_breakdown"] == "Token usage: unreported"
    # Later known requests cannot turn an incomplete cumulative subtotal into
    # a complete total. The tooltip qualifier follows all preceding receipts.
    run.steps.append(run.steps[0].model_copy(update={
        "id": "known", "start_ns": 2, "end_ns": 3,
        "usage": Usage(input_tokens=10, output_tokens=1),
    }))
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert chart["bars"][1]["cumulative_partial"]


def test_previous_context_follows_post_call_and_compaction_baselines(prices):
    """Use the prior context point, replacing it after a completed compaction.

    Cache categories describe billing on the current request; the previous
    context line must come from this same history's earlier context evidence.
    """
    from reporting.render import conversation_turns, cost_chart

    prices.exchange = ExchangeRate(rate="0.8", date="2026-09-20")
    run = Run(id="history", title="Context carry-over", status="ok", steps=[
        Step(id="agent", name="responder", kind="workflow", start_ns=0, end_ns=9,
             status="ok"),
        Step(id="first", parent_id="agent", name="model", kind="model",
             start_ns=1, end_ns=2, status="ok", provider="demo",
             model="scripted-chat", context={"report_history_id": "shared"},
             usage=Usage(input_tokens=100, output_tokens=20, cache_write=90)),
        Step(id="compact", parent_id="agent", name="SummarizationMiddleware.before_model",
             kind="workflow", start_ns=3, end_ns=4, status="ok", context={
                 "compaction_event": "completed",
                 "compaction_before_tokens": 120,
                 "compaction_after_tokens": 40,
                 "compaction_before_basis": "reported input + estimated retained output",
                 "compaction_after_basis": "local estimate after history replacement",
                 "compaction_trigger_tokens": 100,
                 "compaction_max_input_tokens": 500,
             }),
        Step(id="second", parent_id="agent", name="model", kind="model",
             start_ns=5, end_ns=6, status="ok", provider="demo",
             model="scripted-chat", context={"report_history_id": "shared"},
             usage=Usage(input_tokens=60, output_tokens=10)),
        Step(id="third", parent_id="agent", name="model", kind="model",
             start_ns=7, end_ns=8, status="ok", provider="demo",
             model="scripted-chat", context={"report_history_id": "shared"},
             usage=Usage(input_tokens=95, output_tokens=5)),
    ])
    chart = cost_chart(conversation_turns(run, prices), prices)
    first, second, third = chart["bars"]
    description = chart["compactions"][0]["description"]
    assert "Context before: 120 tokens" in description
    assert "History after: 40 tokens" in description
    assert "Trigger: 100 context tokens" in description
    assert "error limit" not in description.lower()
    assert "Change:" not in description
    assert "(" not in description
    assert "History: " not in description
    assert "estimated" not in description.lower()
    assert "Previous context: 0 tokens" in first["token_breakdown"]
    assert "Cache write: 90 tokens" in first["token_breakdown"]
    assert "• Fresh input: 100 tokens" in first["token_breakdown"]
    assert first["context_tokens"] == 120
    assert "Previous context: 40 tokens (after compaction)" in second["token_breakdown"]
    assert "Previous context: 70 tokens (prior post-call)" in third["token_breakdown"]
    # Context growth and cache billing are separate decompositions of input.
    # Fresh input can include all previous context; it is never an addition.
    assert "Net input change: +20 tokens" in second["token_breakdown"]
    assert "Net input change: +25 tokens" in third["token_breakdown"]
    assert "• Previous context" not in third["token_breakdown"]
    assert "Uncached input" not in third["token_breakdown"]


def test_input_composition_is_separate_from_cache_billing(prices):
    """R7's smaller role envelope must reconcile independently of cache reuse."""
    from reporting.render import conversation_turns, cost_chart

    prices.exchange = ExchangeRate(rate="0.8", date="2026-09-20")
    step = Step(id="r7", name="model", kind="model", start_ns=1, end_ns=2,
                status="ok", provider="demo", model="scripted-chat",
                context={"simulated_system_tokens": 1107, "simulated_definitions_tokens": 993},
                usage=Usage(input_tokens=5948, output_tokens=73, cache_read=100))
    run = Run(id="composition", title="Input composition", status="ok", steps=[step])
    tip = cost_chart(conversation_turns(run, prices), prices)["bars"][0]["token_breakdown"]
    assert "• System prompt: 1,107 tokens" in tip
    assert "• Tool documentation: 993 tokens" in tip
    assert "• Conversation: 3,848 tokens" in tip
    assert "Input billing | • Fresh input: 5,848 tokens | • Cache read: 100 tokens" in tip
    step.context.clear()
    tip = cost_chart(conversation_turns(run, prices), prices)["bars"][0]["token_breakdown"]
    assert "• System prompt: unreported" in tip
    assert "• Tool documentation: unreported" in tip
    assert "• Conversation: unreported" in tip


def test_compaction_threshold_is_visible_before_any_compaction(prices):
    """Recorded policy draws a correctly scaled line even below the threshold."""
    from reporting.render import conversation_turns, cost_chart

    prices.exchange = ExchangeRate(rate="0.8", date="2026-09-20")
    step = Step(id="policy", name="model", kind="model", start_ns=1, end_ns=2,
                status="ok", provider="demo", model="scripted-chat",
                context={"compaction_trigger_tokens": 6500},
                usage=Usage(input_tokens=100, output_tokens=20))
    run = Run(id="threshold", title="Compaction threshold", status="ok", steps=[step])
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert chart["token_max"] > 6500
    assert chart["token_triggers"] == [{"value": 6500, "y": 290 - 6500 / chart["token_max"] * 260}]
    if chart["bars"][0]["context"]:
        assert chart["percent_triggers"][0]["value"] == 6500 / chart["bars"][0]["context"]["capacity"] * 100
    step.context.clear()
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert chart["token_triggers"] == []
    assert chart["percent_triggers"] == []


def test_context_capacity_resolves_explicit_alias(prices):
    """An explicit model alias works without guessing suffixes of unknown models."""
    from reporting.context import context_utilization

    prices.aliases["openai:dated-model"] = "openai:gpt-5.5"
    step = Step(
        id="m",
        name="model",
        kind="model",
        start_ns=0,
        end_ns=1,
        status="ok",
        provider="openai",
        model="dated-model",
        usage=Usage(input_tokens=105_000, output_tokens=0),
    )
    assert context_utilization(step, prices)["percent"] == 10


@pytest.mark.parametrize("before,after", [
    (726, 794),
    (1189, 846),
    (500, 500),
])
def test_compaction_description_reports_counts_without_commentary(before, after):
    """Legacy replacement evidence lists actual counts without an interpretation."""
    from reporting.context import compaction_description

    step = Step(id="compact", name="summary", kind="workflow", start_ns=0,
                end_ns=1, status="ok", context={
                    "compaction_event": "completed",
                    "compaction_before_tokens": before,
                    "compaction_after_tokens": after,
                })
    description = compaction_description(step)
    assert f"History before: {before:,} tokens" in description
    assert f"History after: {after:,} tokens" in description
    assert "error limit" not in description.lower()
    assert "size reduction" not in description


def test_scripted_context_tool_points_show_threshold_crossings(tmp_path):
    """Deterministic tool results reveal crossings independently of saved live runs."""
    import subprocess
    import sys
    from tempfile import TemporaryDirectory

    from reporting.render import conversation_turns, cost_chart, render

    # Published samples use real models and therefore have variable call counts.
    # Generate the authored fixture privately so its exact thresholds remain a
    # useful regression check without constraining future live report refreshes.
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path / "context_budget"
    subprocess.run(
        [sys.executable, "-m", "agent_runtime", "--sample", "context_budget",
         "--demo", "--client", "static", "--out", str(folder),
         "--prices", str(root / "models.json"),
         "--fx-file", str(root / "exchange-rate.json")],
        cwd=root, check=True, capture_output=True, text=True, timeout=60,
    )
    run = Run.model_validate_json((folder / "run.json").read_text())
    prices = load_prices(folder / "prices.json")
    turns = conversation_turns(run, prices)
    chart = cost_chart(turns, prices)
    assert len(chart["bars"]) == 47
    assert len(chart["tool_points"]) == 32
    crossings = [p for p in chart["tool_points"] if p["context_tokens"] > 8500]
    assert [p["context_tokens"] for p in crossings] == [9128, 8674]
    assert chart["token_max"] > 9128
    assert all(p["token_y"] < chart["token_triggers"][0]["y"] for p in crossings)
    # Every review attempt has fresh context, including the rejected first
    # implementation. Review reads never join the planner/worker history.
    isolated = [p for p in chart["tool_points"] if p["history_label"] != "Main context"]
    assert len(isolated) == 10
    assert len({p["history_id"] for p in isolated}) == 4
    assert all(p["history_label"].startswith("Isolated review ") for p in isolated)
    assert all(p["context_tokens"] < 8500 for p in isolated)
    for point in chart["tool_points"]:
        assert any(f"{point['center']},{point['token_y']}" in line["points"] for line in chart["token_lines"])
    with TemporaryDirectory() as temporary:
        output = Path(temporary) / "report.html"
        render(run, prices, output)
        html = output.read_text()
        assert html.count('class="tool-context-point"') == 64
    # Missing model usage cannot manufacture a context baseline for its tools.
    for step in run.steps:
        if step.kind == "model":
            step.usage = None
    assert cost_chart(conversation_turns(run, prices), prices)["tool_points"] == []
