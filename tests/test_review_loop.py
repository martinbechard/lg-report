"""Verify real graph routing, judge feedback, limits, and reporting with offline LLMs.

Scripted verdicts prove execution semantics, not the accuracy of a live judge.
Tests include early approval, persistent rejection, invalid output, and new turns.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from lg_report.agents import evidence_judge, review_author
from lg_report.platform.conversation import Conversation, Request
from lg_report.platform.shared_simulated_model import SharedSimulatedModel
from lg_report.platform.simulated_model import ScriptedChatModel
from lg_report.platform.static_client import StaticClient
from lg_report.report.pricing import cost, load_prices, summarize
from lg_report.report.recording import record_run
from lg_report.report.schema import Run
from lg_report.workflows.review_loop import build_workflow
from samples.review_loop.test_case import (
    FINAL_REVIEW,
    FIRST_DRAFT,
    FIRST_REVIEW,
    REVISED_DRAFT,
    USER_PROMPTS,
    make_simulated_model,
)


def test_feedback_drives_real_second_round_and_report(tmp_path):
    client = StaticClient([Request(USER_PROMPTS[0])])
    graph = build_workflow(make_simulated_model(), first_draft_high_level=True)
    prices = load_prices(Path(__file__).parents[1] / "models.json")
    record_run(
        Conversation(graph, client),
        {},
        tmp_path / "run",
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
        include_output=True,
    )
    result = client.results[0]
    assert result["round"] == 2 and result["outcome"] == "approved"
    assert result["messages"][-1].content == REVISED_DRAFT
    # Internal drafts/reviews are not appended to the user's conversation.
    assert len(result["messages"]) == 2
    assert "high-level overview" in result["author_history"][1].content
    for concern in FIRST_REVIEW["feedback"]:
        assert concern in result["author_history"][-2].content
    assert FIRST_DRAFT in result["judge_history"][0].content
    assert REVISED_DRAFT in result["judge_history"][-2].content
    run = Run.model_validate_json((tmp_path / "run/run.json").read_text())
    calls = sorted(
        [step for step in run.steps if step.kind == "model"],
        key=lambda step: step.start_ns,
    )
    assert len(calls) == 4
    assert [call.request[0]["content"] for call in calls] == [
        review_author.SYSTEM_PROMPT,
        evidence_judge.SYSTEM_PROMPT,
        review_author.SYSTEM_PROMPT,
        evidence_judge.SYSTEM_PROMPT,
    ]
    assert calls[0].usage.cache_read == calls[1].usage.cache_read == 0
    assert calls[2].usage.cache_read > 0 and calls[3].usage.cache_read > 0
    assert summarize(run, prices)["known_cost"] == sum(
        cost(call, prices)[0] for call in calls
    )
    html = (tmp_path / "run/report.html").read_text()
    assert "high-level" in html and "30-second" in html


def model_with_reviews(reviews):
    """Keep author responses available for exactly as many evaluations as requested."""
    return SharedSimulatedModel(
        scripts={
            review_author.SYSTEM_PROMPT: [
                AIMessage(content="Draft proposal") for _ in reviews
            ],
            evidence_judge.SYSTEM_PROMPT: [
                AIMessage(content=json.dumps(review)) for review in reviews
            ],
        }
    )


def test_first_round_can_be_approved():
    result = build_workflow(
        model_with_reviews([FINAL_REVIEW]), first_draft_high_level=True
    ).invoke({"messages": [("user", "Give a brief overview")]})
    assert result["round"] == 1 and result["outcome"] == "approved"


def test_rejection_stops_at_limit_without_claiming_approval():
    result = build_workflow(
        model_with_reviews([FIRST_REVIEW] * 3), max_rounds=3
    ).invoke({"messages": [("user", "Give a concrete plan")]})
    assert result["round"] == 3 and result["outcome"] == "limit_reached"
    assert "NOT approved" in result["messages"][-1].content
    assert FIRST_REVIEW["feedback"][0] in result["messages"][-1].content


def test_invalid_judge_output_is_not_approval():
    model = SharedSimulatedModel(
        scripts={
            review_author.SYSTEM_PROMPT: [AIMessage(content=FIRST_DRAFT)],
            evidence_judge.SYSTEM_PROMPT: [AIMessage(content="looks fine to me")],
        }
    )
    with pytest.raises(ValidationError):
        build_workflow(model).invoke({"messages": [("user", "A plan please")]})
    with pytest.raises(ValidationError):
        evidence_judge.Review(
            verdict="approve", rationale="fine", feedback=["fix this"]
        )
    with pytest.raises(ValidationError):
        evidence_judge.Review(verdict="revise", rationale="vague", feedback=[])


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_round_limit(limit):
    with pytest.raises(ValueError):
        build_workflow(make_simulated_model(), max_rounds=limit)


def test_new_user_turn_starts_new_review_cycle():
    # A global response sequence suffices here: this check is about graph-state
    # reset, not simulated cache accounting across separate review sessions.
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="First answer"),
            AIMessage(content=json.dumps(FINAL_REVIEW)),
            AIMessage(content="Second answer"),
            AIMessage(content=json.dumps(FINAL_REVIEW)),
        ]
    )
    client = StaticClient([Request("First question"), Request("Second question")])
    Conversation(build_workflow(model), client).invoke({}, {})
    assert [result["round"] for result in client.results] == [1, 1]
    assert len(client.results[1]["judge_history"]) == 2
    assert len(client.results[1]["messages"]) == 4
