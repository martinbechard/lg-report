"""Verify token accounting, context growth, and report conversation presentation.

Use deterministic graph runs and fixed arithmetic tariffs so provider price
changes cannot alter expected sums. Checks distinguish cache subsets, reasoning,
and nested spans to catch double-counting as well as misleading display order.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
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
from samples.simple_chat.scripted_run import make_simulated_model as make_chat_model
from samples.thinking_agent.scripted_run import (
    make_simulated_model as make_thinking_model,
)
from samples.tool_chat.scripted_run import make_simulated_model as make_tool_model


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
    assert "Fresh input" in html and "Reasoning" in html


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
        fresh = call.index("Fresh input")
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
    heavy_html = next(
        block
        for block in html.split('<details open class="event')
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
    assert len(chart["context_lines"]) == 2
    assert chart["context_missing"] and chart["context_illustrative"]
    assert chart["bars"][0]["context"]["percent"] == 50
    assert chart["bars"][2]["context_y"] == 290
    assert chart["bars"][3]["context_y"] == 30
    assert chart["context_ticks"][-1]["value"] == 150
    destination = tmp_path / "context.html"
    render(run, prices, destination)
    html = destination.read_text()
    assert 'stroke="#17734b" stroke-width="1.5"' in html
    assert 'stroke="#912c42" stroke-width="3"' in html
    assert "50.000%" in html and "0.000%" in html
    assert "Illustrative context used" in html and "Context %" in html
    assert "gaps are not zero" in html


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
