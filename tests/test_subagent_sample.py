"""Verify that delegated work is captured once and keeps its own message context.

Execute the real parent/task/specialist graph with separate scripted model
instances. Assertions ensure the parent receives the specialist summary rather
than its internal conversation, and that nested model costs reconcile.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path

from fixtures.mock_client import MockClient

from agent_runtime.harness.conversation import Conversation, Request
from agent_runtime.harness.sample_catalog import SampleCatalog
from reporting.execute_runnable import execute_runnable
from reporting.pricing import cost, load_prices, summarize
from reporting.render import agent_activity, conversation_turns
from reporting.schema import Run
from samples.subagent_chat.sample import CONVERSATION

# Expected values are test projections of the authored conversation.
DELEGATED_TASK = CONVERSATION[1]["tool_calls"][0]["args"]["description"]
SPECIALIST_SUMMARY = CONVERSATION[4]["content"]
USER_PROMPTS = [entry["content"] for entry in CONVERSATION if entry["role"] == "client"]


def test_delegation_context_and_costs(tmp_path):
    # Parent, task, and specialist spans must preserve isolated
    # prompts while their nested model costs reconcile with rendered activity.
    """Child work is billed once, nested under task, and summarized to the parent."""
    prices = load_prices(Path(__file__).parents[1] / "models.json")
    output = tmp_path / "run"
    execute_runnable(
        Conversation(
            SampleCatalog().create_run("subagent_chat", False)[0],
            MockClient([Request(p) for p in USER_PROMPTS]),
        ),
        {},
        output,
        prices,
        provider="demo",
        model="scripted-chat",
        demo=True,
        include_output=True,
    )
    run = Run.model_validate_json((output / "run.json").read_text())
    models = sorted(
        [s for s in run.steps if s.kind == "model"], key=lambda s: s.start_ns
    )
    assert len(models) == 4 and all(s.status == "ok" for s in run.steps)
    task = next(s for s in run.steps if s.name == "task")
    by_id = {s.id: s for s in run.steps}
    for model in models[1:3]:
        node = model
        while node.id != task.id:
            node = by_id[node.parent_id]
    assert models[1].request[-1]["content"] == DELEGATED_TASK
    assert USER_PROMPTS[0] not in str(models[1].request)
    # The task tool executes the child graph and returns its summary. That
    # tool result becomes the final parent model request, while the child
    # lookup messages remain inside the specialist conversation.
    assert models[3].request[-1]["content"] == SPECIALIST_SUMMARY
    assert "specialist-lookup-1" not in str(models[3].request)
    assert models[0].usage.cache_read == models[1].usage.cache_read == 0
    assert models[2].usage.cache_read > 0 and models[3].usage.cache_read > 0
    assert summarize(run, prices)["known_cost"] == sum(
        cost(s, prices)[0] for s in models
    )

    activities = agent_activity(run, prices)["activities"]
    assert len(activities) == 2
    assert [a["model_calls"] for a in activities] == [2, 2]
    assert activities[1]["caller"].id == activities[0]["step"].id
    assert activities[1]["step"].name == "isolated-subagent"
    assert sum(a["total"] for a in activities) == summarize(run, prices)["known_cost"]
    events = [
        e
        for t in conversation_turns(run, prices)
        for e in t["events"]
        if e["step"].kind == "model"
    ]
    assert [e["agent"].id for e in events] == [
        activities[0]["step"].id,
        activities[1]["step"].id,
        activities[1]["step"].id,
        activities[0]["step"].id,
    ]
    html = (output / "report.html").read_text()
    assert "Agent activity" in html and "Delegation ·" in html

    # Conversation containment follows execution: parent -> task -> specialist.
    turn = conversation_turns(run, prices)[0]
    root = turn["conversation_nodes"][0]
    assert root["step"].id == activities[0]["step"].id
    task_node = next(n for n in root["children"] if n["step"].name == "task")
    child_node = task_node["children"][0]
    assert child_node["step"].id == activities[1]["step"].id
    assert [n["step"].kind for n in child_node["children"]] == [
        "model",
        "tool",
        "model",
    ]
    assert root["children"][-1]["event"]["step"].id == models[-1].id
    assert "conversation-thread" in html and "conversation-children" in html
