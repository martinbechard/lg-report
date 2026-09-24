"""Exercise real nested graphs and inspect their actual model requests.

Run inside the lg-report project environment. Missing framework dependencies
skip this module explicitly; policy-only checks must not be presented as proof
that LangGraph wiring or report rendering works.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Framework imports intentionally follow importorskip below.

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="Requires the lg-report LangGraph environment")
pytest.importorskip("langchain", reason="Requires the lg-report LangChain environment")

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from agent_runtime.workflows.nested_workflows import build_workflow
from samples.nested_workflows.scenarios import USER_PROMPTS
from samples.nested_workflows.scripted_run import build_models


@pytest.mark.parametrize(
    "sample,outcome",
    [
        ("nested_workflows", "COMPLETED"),
        ("nested_workflows_review_limit", "BLOCKED"),
        ("nested_workflows_test_limit", "BLOCKED"),
    ],
)
def test_catalog_cli_report_preserves_context_scopes(tmp_path, sample, outcome):
    """Exercise discovery, factory resolution, tracing and rendering together.

    Run outside the checkout to catch accidental relative-path dependencies.
    Demo mode and saved pricing keep this check independent of provider access.
    A blocked teaching scenario is a successful execution with a blocked output.
    """
    from reporting.pricing import load_prices
    from reporting.render import agent_activity, conversation_turns, cost_chart
    from reporting.schema import Run

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runtime",
            "--sample",
            sample,
            "--demo",
            "--prices",
            str(root / "models.json"),
            "--fx-file",
            str(root / "exchange-rate.json"),
            "--out",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    run = Run.model_validate_json((tmp_path / "run.json").read_text())
    assert run.demo and run.status == "ok"
    assert run.output.startswith(outcome)
    prices = load_prices(tmp_path / "prices.json")
    chart = cost_chart(conversation_turns(run, prices), prices)
    assert not chart["compactions"]
    histories = chart["histories"]
    assert {h["label"] for h in histories} == {
        "Main context",
        "Coding context",
        "Isolated review context #1",
        "Isolated review context #2",
        "Isolated review context #3",
    }
    assert len({h["color"] for h in histories}) == 5
    assert all(
        len(h["bars"]) == 1 for h in histories if h["label"].startswith("Isolated")
    )
    models = [step for step in run.steps if step.kind == "model"]
    assert {step.context["report_context_depth"] for step in models} == {1, 2, 3}
    if sample == "nested_workflows":
        assert len(models) == 18
    html = (tmp_path / "report.html").read_text()
    assert "Coding context" in html
    assert html.index('id="workflow-state-heading"') < html.index(
        'id="collaboration-heading"'
    )
    roles = {
        activity["step"].name for activity in agent_activity(run, prices)["activities"]
    }
    assert roles == {"planner", "coding_supervisor", "coder", "reviewer"} | (
        set() if sample.endswith("review_limit") else {"tester"}
    )
    # Persist the topology only once, and preserve untaken decisions as well
    # as the actual nested-call boundaries. Re-rendering needs no graph code.
    definitions = [
        step.context["report_workflow_definition"]
        for step in run.steps
        if "report_workflow_definition" in step.context
    ]
    assert len(definitions) == 1
    definition = json.loads(definitions[0])
    assert {
        edge["label"] for edge in definition["edges"] if edge["source"] == "planner"
    } == {
        "code",
        "test",
        "finish",
        "escalate",
    }
    coding = definition["children"]["coding_workflow"]
    assert coding["children"]["isolated_review"]["nodes"] == [
        "__start__",
        "reviewer",
        "__end__",
    ]


def build(scenario="rework", **options):
    settings = {"scenario": scenario, **options}
    models = build_models(settings)
    graph = build_workflow(
        planner_model=models["planner"],
        supervisor_model=models["coding_supervisor"],
        coder_model=models["coder"],
        reviewer_model=models["reviewer"],
        tester_model=models["tester"],
        **settings,
    )
    return graph, models


def text(messages):
    return "\n".join(str(message.content) for message in messages)


def run(scenario="rework", **options):
    graph, models = build(scenario, **options)
    result = graph.invoke({"messages": [HumanMessage(content=USER_PROMPTS[0])]})
    return result, models


def test_default_real_graph_and_context_boundaries():
    result, models = run()
    assert result["outcome"] == "completed"
    assert result["coding_cycles"] == 2
    assert result["coding_memory"]["total_rounds"] == 3
    outer = text(result["messages"])
    coding = text(result["coding_memory"]["history"])
    assert "OUTER_ONLY_DETAIL" in outer and "CODING_ONLY_DETAIL" not in outer
    assert "CODING_ONLY_DETAIL" in coding and "OUTER_ONLY_DETAIL" not in coding
    for role in ("planner", "tester"):
        for request in models[role].requests:
            assert "OUTER_ONLY_DETAIL" in text(request)
            assert "CODING_ONLY_DETAIL" not in text(request)
            assert "REVIEW_ONLY_DETAIL" not in text(request)
    for role in ("coding_supervisor", "coder", "reviewer"):
        for request in models[role].requests:
            assert "OUTER_ONLY_DETAIL" not in text(request)


def test_review_is_fresh_each_time():
    _, models = run()
    requests = models["reviewer"].requests
    assert len(requests) == 3
    for index, request in enumerate(requests, start=1):
        assert len(request) == 2  # Current role's system prompt + one explicit packet.
        assert "CODING_ONLY_DETAIL" not in text(request)
        packet = json.loads(request[-1].content)
        assert set(packet) == {"task", "artifact", "reported_defects"}
        assert packet["artifact"]["candidate_id"] == f"candidate-{index}"
        assert "working_notes" not in packet["artifact"]


def test_coding_history_survives_tester_reentry():
    _, models = run()
    repair_request = text(models["coder"].requests[2])
    assert "candidate-1" in repair_request and "candidate-2" in repair_request
    assert "CODING_ONLY_DETAIL" in repair_request
    assert "remove blank tags" in repair_request


def test_inner_limit_never_calls_tester():
    result, models = run("review_limit")
    assert result["outcome"] == "blocked"
    assert result["coding_result"]["status"] == "blocked"
    assert len(models["coder"].requests) == 3
    assert not models["tester"].requests


def test_outer_limit_stops_reentry():
    result, models = run("test_limit")
    assert result["outcome"] == "blocked" and result["coding_cycles"] == 3
    assert result["test_result"]["verdict"] == "fail"
    assert len(models["tester"].requests) == 3
    assert len(models["coder"].requests) == 3


def test_pass_on_last_allowed_round_and_cycle():
    result, _ = run("first_pass", max_review_rounds=1, max_coding_cycles=1)
    assert result["outcome"] == "completed"


def test_async_path():
    graph, models = build()
    result = asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content=USER_PROMPTS[0])]})
    )
    assert result["outcome"] == "completed"
    assert len(models["reviewer"].requests) == 3


def test_trace_context_ids_distinguish_scopes():
    class Capture(BaseCallbackHandler):
        def __init__(self):
            self.scopes = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.scopes.append((kwargs.get("metadata") or {}).get("report_history_id"))

    graph, _ = build()
    capture = Capture()
    graph.invoke(
        {"messages": [HumanMessage(content=USER_PROMPTS[0])]},
        config={"callbacks": [capture]},
    )
    assert len(capture.scopes) == 18
    assert len({s for s in capture.scopes if s.startswith("outer:")}) == 1
    assert len({s for s in capture.scopes if s.startswith("coding:")}) == 1
    assert len({s for s in capture.scopes if s.startswith("review:")}) == 3
