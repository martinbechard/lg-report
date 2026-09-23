"""Exercise role contracts independently of workflow prompt and protocol knowledge.

These offline tests inspect actual messages and validated return values. They
prove that callers supply task data while agents own message framing and parsing;
they do not claim that scripted author/judge/quote decisions are good reasoning.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

import pytest
from langchain.agents.structured_output import MultipleStructuredOutputsError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from agent_runtime.agents import evidence_judge, quote_interpreter, review_author
from agent_runtime.harness.simulated_model import ScriptedChatModel


def test_author_frames_initial_and_revision_tasks_without_workflow_prompts():
    """A caller can request drafts through data and retain history without editing it."""
    agent = review_author.build_agent(
        {
            "model": ScriptedChatModel(
                responses=[
                    AIMessage(content="Initial draft"),
                    AIMessage(content="Revised draft"),
                ]
            )
        }
    )
    conversation = [HumanMessage(content="Explain the evidence")]
    initial = agent.invoke(
        {
            "conversation": conversation,
            "history": [],
            "review": None,
            "high_level": True,
        }
    )
    # The role, not the caller, adds the teaching instruction and revision prose.
    assert "high-level overview" in initial["history"][1].content
    review = {
        "verdict": "revise",
        "rationale": "Missing source",
        "feedback": ["Cite the source"],
    }
    revised = agent.invoke(
        {
            "conversation": conversation,
            "history": initial["history"],
            "review": review,
            "high_level": True,
        }
    )
    assert revised["draft"] == "Revised draft"
    assert json.dumps(review) in revised["history"][-2].content
    # Context retention is a harness decision: agents return new lists and never
    # edit an earlier result or append their system instructions into its history.
    assert len(conversation) == 1
    assert len(initial["history"]) == 3
    assert all(m.type != "system" for m in revised["history"])


def test_judge_owns_evidence_framing_and_returns_validated_review():
    """A workflow can route on the role result without knowing JSON message formats."""
    verdict = {"verdict": "approve", "rationale": "Supported", "feedback": []}
    agent = evidence_judge.build_agent(
        {
            "model": ScriptedChatModel(
                responses=[
                    AIMessage(content=json.dumps(verdict)),
                ]
            )
        }
    )
    result = agent.invoke(
        {
            "conversation": [HumanMessage(content="Explain the supplied evidence")],
            "history": [],
            "draft": "Candidate answer",
        }
    )
    assert result["review"] == verdict
    assert "Candidate answer" in result["history"][0].content
    assert "Explain the supplied evidence" in result["history"][0].content


def test_judge_rejects_invalid_output_before_workflow_routing():
    """Malformed provider text cannot escape the agent as an approval-like result."""
    agent = evidence_judge.build_agent(
        {
            "model": ScriptedChatModel(
                responses=[
                    AIMessage(content="probably approved"),
                ]
            )
        }
    )
    with pytest.raises(ValidationError):
        agent.invoke({"conversation": [], "history": [], "draft": "Candidate"})


@pytest.mark.parametrize(
    "calls",
    [
        [],
        [
            {"name": "WrongDecision", "args": {}, "id": "wrong"},
        ],
        [
            {"name": "QuoteDecision", "args": {}, "id": "one"},
            {"name": "QuoteDecision", "args": {}, "id": "two"},
        ],
    ],
)
def test_quote_agent_rejects_ambiguous_or_wrong_protocol(calls):
    """The human-loop harness must not inspect tool names or select among decisions."""
    agent = quote_interpreter.build_agent(
        {
            "model": ScriptedChatModel(
                responses=[
                    AIMessage(content="", tool_calls=calls),
                ]
            )
        }
    )
    # Native structured output rejects ambiguity; repeated unknown tool calls
    # exhaust LangGraph's explicit test budget rather than becoming a decision.
    error = GraphRecursionError if len(calls) < 2 else MultipleStructuredOutputsError
    with pytest.raises(error):
        agent.invoke(
            {"initial_request": {}, "conversation": []}, {"recursion_limit": 6}
        )


def test_quote_agent_returns_decision_not_provider_tool_calls():
    """A direct caller gets the same validated role contract used by the workflow."""
    decision = {"action": "ask", "reason": "Ambiguous units", "text": "Metres or feet?"}
    agent = quote_interpreter.build_agent(
        {
            "model": ScriptedChatModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "QuoteDecision",
                                "args": decision,
                                "id": "decision",
                            }
                        ],
                    ),
                ]
            )
        }
    )
    result = agent.invoke({"initial_request": {"length": 10}, "conversation": []})
    assert isinstance(result, quote_interpreter.QuoteDecision)
    assert result.model_dump() == decision


@pytest.mark.parametrize("role", ["author", "judge", "quote"])
def test_role_runs_workflow_middleware(role):
    """The plain parameter dictionary must carry working middleware into every role."""
    from langchain.agents.middleware import AgentMiddleware

    calls = []

    class ObserveModel(AgentMiddleware):
        """Observe the actual model boundary, rather than only constructor arguments."""

        def before_model(self, state, runtime):
            """Record that the native graph applied workflow-supplied middleware."""
            calls.append(state["messages"])

    if role == "author":
        builder = review_author.build_agent
        request = {
            "conversation": [],
            "history": [],
            "review": None,
            "high_level": True,
        }
        response = AIMessage(content="A draft")
    elif role == "judge":
        builder = evidence_judge.build_agent
        request = {"conversation": [], "history": [], "draft": "A draft"}
        response = AIMessage(
            content=json.dumps(
                {"verdict": "approve", "rationale": "Supported", "feedback": []}
            )
        )
    else:
        builder = quote_interpreter.build_agent
        request = {"initial_request": {}, "conversation": []}
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "QuoteDecision",
                    "id": "decision",
                    "args": {
                        "action": "ask",
                        "reason": "Missing scope",
                        "text": "What scope?",
                    },
                }
            ],
        )
    agent = builder(
        {
            "model": ScriptedChatModel(responses=[response]),
            "middleware": [ObserveModel()],
        }
    )
    agent.invoke(request)
    assert len(calls) == 1
