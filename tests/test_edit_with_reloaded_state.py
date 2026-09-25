"""Verify authoritative claim state and exact model context independently of answers.

Scripted answers cannot establish a live accuracy improvement. These tests prove
purge timing, read freshness, complete tool exchanges, failed-edit behavior, and
audit retention using the real agent/tool loop without a provider request.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

import pytest
from fixtures.mock_client import MockClient
from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime.agents.claims_agent import build_agent, invalidate_claim_context
from agent_runtime.harness.conversation import Conversation, Request
from agent_runtime.harness.simulated_model import SimulatedModel
from agent_runtime.tools.claims import ClaimStore
from agent_runtime.workflows.edit_with_reloaded_state import (
    ClaimsContext,
    ContextAudit,
)
from samples.edit_with_reloaded_state.sample import CONVERSATION, make_simulated_model

# Expected values are test projections of the authored conversation.
CORRECTED_DESCRIPTION = next(
    call["args"]["description"]
    for entry in CONVERSATION
    for call in entry.get("tool_calls", [])
    if call["name"] == "edit_claim"
)
USER_PROMPTS = [entry["content"] for entry in CONVERSATION if entry["role"] == "client"]


def run_demo(mode, *, prompts=USER_PROMPTS, model=None, capture=True):
    """Make each assertion start with an isolated store, client, and model cursor."""
    client = MockClient([Request(prompt) for prompt in prompts])
    audit = ContextAudit(capture_content=capture, write=lambda _: None)
    session = ClaimsContext(
        build_agent({"model": model or make_simulated_model(mode)}),
        mode,
        audit=audit,
        write=lambda _: None,
    )
    Conversation(session.graph, client).invoke({}, {})
    return session, client


@pytest.mark.parametrize(
    "mode", ["edit-with-patched-state", "edit-with-reloaded-state"]
)
def test_same_edits_different_actual_model_context(mode):
    """Compare submitted inputs rather than trusting the authored final answers."""
    session, client = run_demo(mode)
    assert (
        session.agent.evidence()["final_claim"]["description"] == CORRECTED_DESCRIPTION
    )
    assert session.agent.evidence()["final_claim"]["status"] == "approved"
    assert session.agent.context_version == 2
    final_input = session.audit.calls[-1]["messages"]
    serialized = json.dumps(final_input)
    if mode == "edit-with-patched-state":
        assert "parked car" in serialized
        assert '"edit_claim"' in serialized
        # The only read result is revision 1; edits do not insert a new snapshot.
        observations = [m for m in final_input if m["role"] == "tool"]
        assert json.loads(observations[1]["content"])["revision"] == 1
        assert json.loads(observations[2]["content"]) == {"ok": True, "revision": 2}
        assert session.events == []
    else:
        assert "parked car" not in serialized
        assert "stolen" not in serialized
        assert '"edit_claim"' not in serialized
        observations = [m for m in final_input if m["role"] == "tool"]
        assert len(observations) == 2
        assert json.loads(observations[1]["content"])["revision"] == 2
        assert (
            json.loads(observations[1]["content"])["description"]
            == CORRECTED_DESCRIPTION
        )
        # Both read calls were emitted by the model, not the application.
        calls = [c for m in final_input for c in m["tool_calls"]]
        assert [c["id"] for c in calls] == ["policy-read", "follow-up-read"]
        assert calls[1]["id"] == observations[1]["tool_call_id"]
        assert [e["event"] for e in session.events] == ["purge"]
        # Purging working history must not rewrite already saved result/audit data.
        assert "parked car" in json.dumps(session.audit.calls[3])
        assert "parked car" in str(client.results[2]["messages"])


def test_quit_after_edit_does_not_reload():
    """Reload is demand-driven; a completed edit followed by exit needs no read."""
    session, _ = run_demo("edit-with-reloaded-state", prompts=USER_PROMPTS[:3])
    assert "parked car" not in str(session.history)
    assert [
        m.tool_calls[0]["name"] for m in session.history if getattr(m, "tool_calls", [])
    ] == ["read_policy"]
    assert [e["event"] for e in session.events] == ["purge"]
    assert len(session.audit.calls) == 6


@pytest.mark.parametrize(
    "mode", ["edit-with-patched-state", "edit-with-reloaded-state"]
)
def test_multiple_edits_in_one_turn_reload_the_combined_result(mode):
    """Every successful replacement contributes, including across tool calls.

    The first edit changes both fields. The second changes the description again
    while preserving the new status. Execute calls sequentially so each uses the
    revision returned by the preceding edit, as a real agent must do.
    """
    model = make_simulated_model(mode)
    final_description = "The damaged laptop was repaired at the local service centre."
    insertion = [
        i
        for i, entry in enumerate(model.conversation)
        if entry["role"] == "claims_agent"
    ][5]
    model.conversation[insertion:insertion] = [
        {
            "role": "claims_agent",
            "content": response.content,
            "tool_calls": response.tool_calls,
        }
        for response in [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_claim",
                        "id": "second-correction",
                        "args": {
                            "expected_revision": 2,
                            "description": final_description,
                            "status": "approved",
                        },
                    }
                ],
            ),
        ]
    ]
    session, _ = run_demo(mode, model=model)
    assert session.agent.context_version == 3
    assert session.agent.evidence()["final_claim"]["description"] == final_description
    assert session.agent.evidence()["final_claim"]["status"] == "approved"
    final_input = session.audit.calls[-1]["messages"]
    if mode == "edit-with-reloaded-state":
        reads = [m for m in final_input if m["role"] == "tool"]
        assert len(reads) == 2
        assert json.loads(reads[1]["content"]) == {
            "claim_id": "CLM-001",
            "revision": 3,
            "description": final_description,
            "status": "approved",
        }
        assert "parked car" not in json.dumps(final_input)
        assert CORRECTED_DESCRIPTION not in json.dumps(final_input)
        assert [e["event"] for e in session.events] == ["purge"]
    else:
        # The model must combine the original read and BOTH edits; no refreshed
        # snapshot has been silently substituted by the workflow.
        calls = [c for m in final_input for c in m["tool_calls"]]
        assert [c["name"] for c in calls] == [
            "read_policy",
            "read_claim",
            "edit_claim",
            "edit_claim",
        ]
        assert "parked car" in json.dumps(final_input)
        assert session.events == []


def test_failed_edit_preserves_claim_and_history():
    """Revision conflicts return a real error without purging useful evidence."""
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_claim",
                    "args": {},
                    "id": "read",
                }
            ],
        ),
        AIMessage(content="Original claim read."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "edit_claim",
                    "id": "bad-edit",
                    "args": {
                        "expected_revision": 0,
                        "description": "wrong",
                        "status": "approved",
                    },
                }
            ],
        ),
        AIMessage(content="The edit failed."),
        AIMessage(content="The original claim still applies."),
    ]
    session, _ = run_demo(
        "edit-with-reloaded-state",
        prompts=USER_PROMPTS[1:4],
        model=SimulatedModel(
            cache_reuse=False,
            conversation=[
                {
                    "role": "claims_agent",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
                for response in responses
            ],
        ),
    )
    assert session.agent.context_version == 1
    assert session.events == []
    assert "parked car" in json.dumps(session.audit.calls[-1])
    assert "Stale revision" in json.dumps(session.audit.calls[-1])


@pytest.mark.parametrize("description,status", [("", "approved"), ("ok", "unknown")])
def test_invalid_edit_is_atomic(description, status):
    """Invalid fields cannot leave a partially updated authoritative record."""
    store = ClaimStore()
    before = store.claim
    result = json.loads(
        store.tools[1].invoke(
            {
                "expected_revision": 1,
                "description": description,
                "status": status,
            }
        )
    )
    assert result["ok"] is False
    assert store.claim == before


def test_metadata_only_omits_claim_and_prompt_content():
    """The extra audit must honor the same content boundary as standard reports."""
    session, _ = run_demo("edit-with-reloaded-state", capture=False)
    evidence = session.evidence()
    assert "final_claim" not in evidence
    assert all("messages" not in call for call in evidence["calls"])
    assert "laptop" not in json.dumps(evidence)


def test_no_edit_keeps_history_without_reload():
    """edit-with-reloaded-state mode preserves normal continuity until an actual edit succeeds."""
    model = SimulatedModel(
        cache_reuse=False,
        conversation=[
            {
                "role": "claims_agent",
                "content": response.content,
                "tool_calls": response.tool_calls,
            }
            for response in [AIMessage(content="Hello"), AIMessage(content="Hi")]
        ],
    )
    session, _ = run_demo(
        "edit-with-reloaded-state", prompts=["Hello", "Again"], model=model
    )
    assert session.events == []
    assert "Hello" in json.dumps(session.audit.calls[-1])


def test_policy_follow_up_does_not_reload_claim_after_edit():
    """Removing a claim does not trigger retrieval on an unrelated next question."""
    session, _ = run_demo("edit-with-reloaded-state", prompts=USER_PROMPTS[:4])
    call = session.audit.calls[-1]
    assert call["turn"] == 4
    text = json.dumps(call["messages"])
    assert "parked car" not in text
    assert CORRECTED_DESCRIPTION not in text
    tool_calls = [c for m in call["messages"] for c in m["tool_calls"]]
    assert [c["name"] for c in tool_calls] == ["read_policy"]
    assert len(session.audit.calls) == 7
    assert session.events[0]["retained_tool_results"] == 1


def test_no_read_happens_unless_model_requests_one():
    """No preload and no hidden retrieval even when a question mentions a claim."""
    model = SimulatedModel(
        cache_reuse=False,
        conversation=[
            {
                "role": "claims_agent",
                "content": response.content,
                "tool_calls": response.tool_calls,
            }
            for response in [AIMessage(content="Which detail do you need?")]
        ],
    )
    session, _ = run_demo(
        "edit-with-reloaded-state", prompts=["Help with my claim"], model=model
    )
    assert len(session.audit.calls) == 1
    first = session.audit.calls[0]["messages"]
    assert all(m["role"] != "tool" for m in first)
    assert "parked car" not in json.dumps(first)
    assert "deductible_cad" not in json.dumps(first)


def test_mixed_tool_batch_retains_only_actual_policy_pair():
    """Pruning a batch never invents a call or leaves an unmatched tool result."""
    original = [
        AIMessage(
            content="Old claim prose",
            tool_calls=[
                {"name": "read_claim", "args": {}, "id": "c"},
                {"name": "read_policy", "args": {}, "id": "p"},
            ],
        ),
        ToolMessage(content="Old claim", tool_call_id="c"),
        ToolMessage(content="Actual policy", tool_call_id="p"),
    ]
    retained = invalidate_claim_context(original)
    assert retained[1].tool_calls == [original[0].tool_calls[1]]
    assert retained[2] is original[2]
    assert "Old claim" not in str(retained)
    assert original[0].content == "Old claim prose"


def test_agent_can_read_claim_without_workflow_or_policy_loading():
    """The named agent owns tool use independently of conversation orchestration.

    Supply a fresh human question directly to the agent. Its model requests only
    the claim; no workflow is present to select or execute an application read.
    """
    model = SimulatedModel(
        cache_reuse=False,
        conversation=[
            {
                "role": "claims_agent",
                "content": response.content,
                "tool_calls": response.tool_calls,
            }
            for response in [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_claim",
                            "args": {},
                            "id": "claim-only",
                        }
                    ],
                ),
                AIMessage(content="The claim is pending."),
            ]
        ],
    )
    result = build_agent({"model": model}, ClaimStore()).invoke(
        {
            "messages": [{"role": "user", "content": "What is the claim's status?"}],
        }
    )
    observations = [m for m in result["messages"] if m.type == "tool"]
    assert len(observations) == 1
    assert observations[0].tool_call_id == "claim-only"
    assert json.loads(observations[0].content)["status"] == "pending"
    assert "deductible_cad" not in str(result["messages"])


@pytest.mark.parametrize(
    "mode", ["edit-with-patched-state", "edit-with-reloaded-state"]
)
def test_harness_controls_strategy_through_agent_interface(mode):
    """Prove the harness can manage an agent with no claim store or tool names.

    This probe models only the public contract. It exposes an opaque version and
    a valid-context projection; the harness must choose whether to apply it.
    """

    class ProbeAgent:
        """Stand in for another domain whose private implementation is unknown."""

        def __init__(self):
            # The harness may observe this signal but has no domain schema to
            # inspect. Keep evidence of its calls for assertions below.
            self.context_version = 0
            self.inputs = []
            self.projections = 0

        def invoke(self, inputs, config=None):
            # Simulate a domain change on the first turn. Return ordinary
            # messages so history handling uses the real harness implementation.
            self.inputs.append(list(inputs["messages"]))
            if len(self.inputs) == 1:
                self.context_version += 1
            return {"messages": [*inputs["messages"], AIMessage(content="Completed")]}

        def retain_unaffected_context(self, messages):
            # This domain declares none of its old context valid. It does not
            # choose when to run: only the harness can request the projection.
            self.projections += 1
            return []

        def evidence(self):
            # Reporting is also encapsulated; no claim-specific field is needed.
            return {"domain_version": self.context_version}

    agent = ProbeAgent()
    client = MockClient([Request("Change something"), Request("Next question")])
    session = ClaimsContext(
        agent,
        mode,
        write=lambda _: None,
    )
    Conversation(session.graph, client).invoke({}, {})
    assert agent.projections == (1 if mode == "edit-with-reloaded-state" else 0)
    assert len(agent.inputs[1]) == (1 if mode == "edit-with-reloaded-state" else 3)
    assert session.evidence()["domain_version"] == 1
