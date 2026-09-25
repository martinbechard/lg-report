"""Verify parent dispatch, nested review execution, and portable report topology.

Scripted model replies exercise the real graphs without asserting live judgment
quality. Checks cover revision, exhaustion, async execution, and fresh assignments.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agent_runtime.harness.simulated_model import ScriptedChatModel, SimulatedModel
from agent_runtime.workflows.nested_workflows import build_workflow
from samples.nested_workflows.sample import CONVERSATION


def build(**options):
    """Give each invocation an independent cursor over the authored exchange."""
    return build_workflow(SimulatedModel(conversation=CONVERSATION), **options)


def payload():
    """Use the same client request as catalog-driven execution."""
    return {"messages": [("user", CONVERSATION[0]["content"])]}


def test_parent_returns_revised_child_result():
    """The rejected first draft must be replaced before the parent completes."""
    result = build().invoke(payload())
    assert result["assignment"] == CONVERSATION[1]["content"]
    assert result["outcome"] == "approved"
    assert result["messages"][-1].content == CONVERSATION[4]["content"]
    assert len(result["messages"]) == 2


def test_limit_returns_unapproved_draft_and_feedback():
    """Finishing graph execution is not equivalent to receiving approval."""
    result = build(max_rounds=1).invoke(payload())
    assert result["outcome"] == "limit_reached"
    assert "NOT approved" in result["messages"][-1].content
    assert CONVERSATION[2]["content"] in result["answer"]
    for feedback in json.loads(CONVERSATION[3]["content"])["feedback"]:
        assert feedback in result["answer"]


def test_approval_on_last_round():
    """A passing final allowed review must still return approval."""
    assert build(max_rounds=2).invoke(payload())["outcome"] == "approved"


def test_async_parent_executes_child_loop():
    """The same graph supports the asynchronous client entry point."""
    result = asyncio.run(build().ainvoke(payload()))
    assert result["outcome"] == "approved"
    assert result["answer"] == CONVERSATION[4]["content"]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_round_limit(limit):
    """Reject invalid budgets before any role executes."""
    with pytest.raises(ValueError, match="positive integer"):
        build(max_rounds=limit)


def test_invalid_judge_response_propagates():
    """An invalid review cannot become a successful parent result."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Write the function."),
            AIMessage(content="A draft"),
            AIMessage(content="Looks fine"),
        ]
    )
    with pytest.raises(ValidationError):
        build_workflow(model).invoke(payload())


def test_new_assignment_starts_fresh_child_loop():
    """A previous approval must not skip drafting and review on the next turn."""
    approval = CONVERSATION[-1]["content"]
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="First assignment"),
            AIMessage(content="First answer"),
            AIMessage(content=approval),
            AIMessage(content="Second assignment"),
            AIMessage(content="Second answer"),
            AIMessage(content=approval),
        ]
    )
    graph = build_workflow(model, max_rounds=1)
    first = graph.invoke({"messages": [("user", "First request")]})
    second = graph.invoke(
        {**first, "messages": [*first["messages"], ("user", "Second request")]}
    )
    assert second["assignment"] == "Second assignment"
    assert second["answer"] == "Second answer"
    assert second["outcome"] == "approved"


@pytest.mark.parametrize(
    "sample,call_count",
    [
        ("nested_workflows", 5),
        ("nested_workflows_review_limit", 3),
    ],
)
def test_catalog_cli_report_records_nested_execution(tmp_path, sample, call_count):
    """Exercise catalog, role routing, child calls, and saved diagrams end to end."""
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
    calls = sorted(
        [step for step in run.steps if step.kind == "model"], key=lambda s: s.start_ns
    )
    assert len(calls) == call_count
    # Real request order proves the parent dispatched before the child reviewed.
    from agent_runtime.agents import evidence_judge, review_author, work_planner

    assert [call.request[0]["content"] for call in calls] == [
        work_planner.SYSTEM_PROMPT,
        *[
            prompt
            for _ in range((call_count - 1) // 2)
            for prompt in (review_author.SYSTEM_PROMPT, evidence_judge.SYSTEM_PROMPT)
        ],
    ]
    assert CONVERSATION[1]["content"] in str(calls[1].request)
    if call_count == 5:
        assert run.output == CONVERSATION[4]["content"]
        for feedback in json.loads(CONVERSATION[3]["content"])["feedback"]:
            assert feedback in str(calls[3].request)
    else:
        assert "NOT approved" in run.output
    definitions = [
        step.context["report_workflow_definition"]
        for step in run.steps
        if "report_workflow_definition" in step.context
    ]
    assert len(definitions) == 1
    definition = json.loads(definitions[0])
    assert definition["nodes"] == [
        "__start__",
        "planner",
        "review_workflow",
        "finish",
        "__end__",
    ]
    child = definition["children"]["review_workflow"]
    assert {edge["target"] for edge in child["edges"] if edge["source"] == "judge"} == {
        "author",
        "finish",
    }
    html = (tmp_path / "report.html").read_text()
    assert "Child review loop" in html
    assert html.index('id="workflow-state-heading"') < html.index(
        'id="collaboration-heading"'
    )
