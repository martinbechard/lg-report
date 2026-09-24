"""Exercise real nested graphs and inspect their actual model requests.

Run inside the lg-report project environment. Missing framework dependencies
skip this module explicitly; policy-only checks must not be presented as proof
that LangGraph wiring or report rendering works.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Framework imports intentionally follow importorskip below.
# ruff: noqa: E402

import asyncio
import json

import pytest

pytest.importorskip("langgraph", reason="Requires the lg-report LangGraph environment")
pytest.importorskip("langchain", reason="Requires the lg-report LangChain environment")

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from agent_runtime.workflows.nested_workflows import build_workflow
from samples.nested_workflows.scenarios import USER_PROMPTS
from samples.nested_workflows.scripted_run import build_models


def build(scenario="rework", **options):
    settings = {"scenario": scenario, **options}
    models = build_models(settings)
    graph = build_workflow(
        planner_model=models["planner"], supervisor_model=models["coding_supervisor"],
        coder_model=models["coder"], reviewer_model=models["reviewer"],
        tester_model=models["tester"], **settings,
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
    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=USER_PROMPTS[0])]}))
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
    graph.invoke({"messages": [HumanMessage(content=USER_PROMPTS[0])]}, config={"callbacks": [capture]})
    assert len(capture.scopes) == 18
    assert len({s for s in capture.scopes if s.startswith("outer:")}) == 1
    assert len({s for s in capture.scopes if s.startswith("coding:")}) == 1
    assert len({s for s in capture.scopes if s.startswith("review:")}) == 3
