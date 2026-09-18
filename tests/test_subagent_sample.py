"""Check local delegation accounting and context isolation.

AI attribution: Generated with AI assistance.
"""

from pathlib import Path

from lg_report.pricing import cost, load_prices, summarize
from lg_report.runner import ConversationAgent, record_run
from lg_report.schema import Run
from samples.subagent_chat.app import USER_PROMPTS, create_graph
from samples.subagent_chat.simulation import DELEGATED_TASK, SPECIALIST_SUMMARY


def test_delegation_context_and_costs(tmp_path):
    """Child work is billed once, nested under task, and summarized to the parent."""
    prices = load_prices(Path(__file__).parents[1] / "models.json")
    output = tmp_path / "run"
    record_run(
        ConversationAgent(create_graph(False), USER_PROMPTS),
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
    assert models[3].request[-1]["content"] == SPECIALIST_SUMMARY
    assert "specialist-lookup-1" not in str(models[3].request)
    assert models[0].usage.cache_read == models[1].usage.cache_read == 0
    assert models[2].usage.cache_read > 0 and models[3].usage.cache_read > 0
    assert summarize(run, prices)["known_cost"] == sum(
        cost(s, prices)[0] for s in models
    )
