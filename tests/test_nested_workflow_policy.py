"""Verify contracts, finite-loop decisions, and the authored demonstration story.

These tests do not claim to execute LangGraph. The separate integration module
checks real graph wiring and actual model inputs when project dependencies are
installed. Only trusted, locally authored fixture source is executed below.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from collections import deque

import pytest
from pydantic import ValidationError

from agent_runtime.agents.nested_contracts import (
    Assessment,
    Implementation,
    PlannerDecision,
    SupervisorDecision,
)
from agent_runtime.workflows.nested_policy import (
    Limits,
    bind_assessment,
    planner_action,
    require_action,
    supervisor_action,
)
from samples.nested_workflows.scenarios import SOURCE_V1, SOURCE_V2, SOURCE_V3, script


def assessment(verdict="pass", candidate="candidate-1"):
    return {
        "candidate_id": candidate,
        "verdict": verdict,
        "findings": [] if verdict == "pass" else ["Fix the defect."],
        "evidence": "Scripted check.",
        "evidence_kind": "scripted",
    }


def approved():
    return {
        "status": "approved_for_testing",
        "artifact": {"candidate_id": "candidate-1", "source": "pass"},
    }


@pytest.mark.parametrize(
    "rounds,verdict,expected",
    [
        (0, None, "code"),
        (1, "fail", "code"),
        (3, "fail", "escalate"),
        (3, "pass", "return"),
    ],
)
def test_inner_guard(rounds, verdict, expected):
    assert (
        supervisor_action(None if verdict is None else assessment(verdict), rounds, 3)
        == expected
    )


@pytest.mark.parametrize(
    "cycles,verdict,expected",
    [
        (1, "fail", "code"),
        (3, "fail", "escalate"),
        (3, "pass", "finish"),
    ],
)
def test_outer_guard(cycles, verdict, expected):
    assert planner_action(approved(), assessment(verdict), cycles, 3) == expected


def test_review_pass_only_allows_testing():
    assert planner_action(approved(), None, 1, 3) == "test"


def test_inner_block_cannot_reach_tester():
    assert planner_action({"status": "blocked"}, None, 1, 3) == "escalate"


def test_initial_dispatch():
    assert planner_action(None, None, 0, 3) == "code"


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "3", None])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        Limits(value, 3)
    with pytest.raises(ValueError):
        Limits(3, value)


def test_assessment_cannot_approve_with_findings():
    with pytest.raises(ValidationError):
        Assessment.model_validate({**assessment(), "findings": ["Fix this."]})


def test_failure_requires_findings():
    with pytest.raises(ValidationError):
        Assessment.model_validate({**assessment("fail"), "findings": []})


def test_no_fabricated_execution_evidence_kind():
    with pytest.raises(ValidationError):
        Assessment.model_validate({**assessment(), "evidence_kind": "executed"})


def test_stale_candidate_rejected():
    with pytest.raises(ValueError, match="different candidate"):
        bind_assessment(assessment(candidate="candidate-old"), approved()["artifact"])


def test_gate_cannot_be_skipped():
    with pytest.raises(ValueError, match="workflow permits"):
        require_action("finish", "test")


def test_coder_notes_are_not_part_of_handoff():
    response = Implementation(
        candidate_id="candidate-1", source="pass", working_notes="private"
    )
    assert response.handoff().model_dump() == {
        "candidate_id": "candidate-1",
        "source": "pass",
    }


def replay_policy(scenario, max_review_rounds=3, max_coding_cycles=3):
    """Consume each authored decision using the real pure guard/validation code.

    This is a fixture consistency check, not a replacement graph implementation.
    A strict finite upper bound in the test detects errors in this replay itself.
    """
    queues = {
        key: deque(values)
        for key, values in script(
            {
                "scenario": scenario,
                "max_review_rounds": max_review_rounds,
                "max_coding_cycles": max_coding_cycles,
            }
        ).items()
    }
    counts = {key: 0 for key in queues}
    coding = testing = artifact = None
    cycles = attempts = 0

    def take(role, schema):
        counts[role] += 1
        return schema.model_validate(queues[role].popleft()).model_dump()

    for _ in range(3 * max_coding_cycles + 3):
        action = planner_action(coding, testing, cycles, max_coding_cycles)
        require_action(take("planner", PlannerDecision)["action"], action)
        if action in {"finish", "escalate"}:
            assert not any(queues.values()), (
                "Unconsumed decisions would misdescribe the run"
            )
            return action, counts, cycles, attempts
        if action == "test":
            testing = take("tester", Assessment)
            bind_assessment(testing, coding["artifact"])
            continue
        cycles += 1
        review = None
        for rounds in range(max_review_rounds + 1):
            command = supervisor_action(review, rounds, max_review_rounds)
            require_action(
                take("coding_supervisor", SupervisorDecision)["action"], command
            )
            if command != "code":
                coding = {
                    "status": "approved_for_testing"
                    if command == "return"
                    else "blocked",
                    "artifact": artifact,
                }
                testing = None
                break
            attempts += 1
            implementation = Implementation.model_validate(
                take("coder", Implementation)
            )
            artifact = implementation.handoff().model_dump()
            assert artifact["candidate_id"] == f"candidate-{attempts}"
            review = take("reviewer", Assessment)
            bind_assessment(review, artifact)
    raise AssertionError("Fixture replay exceeded its finite bound")


def test_default_story_contains_both_repair_paths():
    action, counts, cycles, attempts = replay_policy("rework")
    assert (action, cycles, attempts) == ("finish", 2, 3)
    assert counts == {
        "planner": 5,
        "coding_supervisor": 5,
        "coder": 3,
        "reviewer": 3,
        "tester": 2,
    }


def test_review_limit_story_does_not_test():
    action, counts, cycles, attempts = replay_policy("review_limit")
    assert (action, cycles, attempts) == ("escalate", 1, 3)
    assert counts["tester"] == 0


def test_test_limit_story_stops_reentry():
    action, counts, cycles, attempts = replay_policy("test_limit")
    assert (action, cycles, attempts) == ("escalate", 3, 3)
    assert counts["tester"] == 3


def test_passing_on_both_final_allowed_attempts_wins():
    assert replay_policy("first_pass", 1, 1)[0] == "finish"


@pytest.mark.parametrize(
    "source,expected",
    [
        (SOURCE_V1, ["", "alpha", "alpha"]),
        (SOURCE_V2, ["", "alpha"]),
        (SOURCE_V3, ["alpha"]),
    ],
)
def test_authored_fixture_defects_are_real(source, expected):
    # These are this sample's trusted constants, never LLM/user-supplied source.
    namespace = {}
    exec(compile(source, "<trusted-demo-fixture>", "exec"), namespace)  # noqa: S102 - trusted authored fixture only
    original = ["  ", "Alpha", "alpha"]
    assert namespace["normalize_tags"](original) == expected
    assert original == ["  ", "Alpha", "alpha"]


def test_fixed_fixture_preserves_order_and_does_not_mutate():
    namespace = {}
    exec(compile(SOURCE_V3, "<trusted-demo-fixture>", "exec"), namespace)  # noqa: S102 - trusted authored fixture only
    original = [" Beta ", "alpha", "BETA", "", " gamma"]
    assert namespace["normalize_tags"](original) == ["beta", "alpha", "gamma"]
    assert original == [" Beta ", "alpha", "BETA", "", " gamma"]
