"""Verify real delegation and architectural boundaries with offline model fixtures.

These checks prove routing-tool execution, isolation, and cost accounting. Fixed
responses do not prove a live model will classify every natural-language question.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import ast
from pathlib import Path

from lg_report.platform.conversation import Conversation, Request
from lg_report.platform.static_client import StaticClient
from lg_report.report.pricing import cost, load_prices, summarize
from lg_report.report.recording import record_run
from lg_report.report.schema import Run
from samples.expert_dispatch.app import create_run
from samples.expert_dispatch.test_case import CASES


def test_expert_selection_isolation_and_costs(tmp_path):
    graph, provider, model = create_run(False)
    client = StaticClient([Request(question) for _, question, _ in CASES])
    prices = load_prices(Path(__file__).parents[1] / "models.json")
    record_run(
        Conversation(graph, client),
        {},
        tmp_path / "run",
        prices,
        provider=provider,
        model=model,
        demo=True,
        include_output=True,
    )
    run = Run.model_validate_json((tmp_path / "run/run.json").read_text())
    models = sorted(
        [s for s in run.steps if s.kind == "model"], key=lambda s: s.start_ns
    )
    tasks = sorted(
        [s for s in run.steps if s.kind == "tool" and s.name == "task"],
        key=lambda s: s.start_ns,
    )
    assert len(models) == 12 and len(tasks) == 3
    by_id = {s.id: s for s in run.steps}
    for i, (expert, question, answer) in enumerate(CASES):
        parent, child, grounded, final = models[i * 4 : i * 4 + 4]
        assert expert in str(parent.response)
        assert child.request[-1]["content"] == question
        assert final.request[-1]["content"] == answer
        assert client.results[i]["messages"][-1].content == answer
        # The selected expert executes beneath the native task span, never as an
        # unrelated top-level call or as three experts answering every question.
        node = child
        ancestors = []
        while node.parent_id:
            node = by_id[node.parent_id]
            ancestors.append(node)
        assert any(s.id == tasks[i].id for s in ancestors)
        assert any(s.name == expert for s in ancestors)
        assert child.usage.cache_read == 0
        assert grounded.usage.cache_read > 0
        assert "Source:" in grounded.request[-1]["content"]
        assert "Source:" in str(grounded.response)
        for _, other_question, _ in CASES[:i]:
            assert other_question not in str(child.request)
    assert summarize(run, prices)["known_cost"] == sum(
        cost(s, prices)[0] for s in models
    )


def test_shared_model_retains_separate_tool_bindings(monkeypatch):
    from lg_report.platform.shared_simulated_model import SharedSimulatedModel
    from lg_report.workflows.expert_dispatch import build_workflow
    from samples.expert_dispatch.test_case import make_simulated_model

    bound_schemas = []
    original = SharedSimulatedModel.bind_tools

    def capture_binding(self, tools, **kwargs):
        binding = original(self, tools, **kwargs)
        bound_schemas.append(binding.kwargs["tool_definitions"])
        return binding

    monkeypatch.setattr(SharedSimulatedModel, "bind_tools", capture_binding)
    model = make_simulated_model()
    Conversation(
        build_workflow(model),
        StaticClient([Request(question) for _, question, _ in CASES]),
    ).invoke({}, {})
    assert all(len(schemas) == 1 for schemas in bound_schemas)
    assert {schemas[0]["function"]["name"] for schemas in bound_schemas} == {
        "task",
        "search_movie_reference",
        "search_sports_reference",
        "search_history_reference",
    }
    task_schema = next(
        schemas for schemas in bound_schemas if schemas[0]["function"]["name"] == "task"
    )
    for expert, _, _ in CASES:
        assert expert in str(task_schema)
    assert "general-purpose" not in str(task_schema)


def test_live_configures_one_model(monkeypatch):
    from samples.expert_dispatch import app

    model_instance = object()
    configured_calls = []

    def configure_once():
        configured_calls.append(True)
        return model_instance, "provider", "model"

    monkeypatch.setattr(app, "configured_model", configure_once)
    monkeypatch.setattr(app, "build_workflow", lambda model: model)
    graph, provider, model = app.create_run(True)
    assert graph is model_instance
    assert len(configured_calls) == 1
    assert (provider, model) == ("provider", "model")


def test_workflow_passes_same_model_to_all_agents(monkeypatch):
    from lg_report.workflows import expert_dispatch as workflow

    model = object()
    received = []

    def build_expert(shared_model):
        received.append(shared_model)
        return object()

    for expert in (
        workflow.movie_expert,
        workflow.sports_expert,
        workflow.history_expert,
    ):
        monkeypatch.setattr(expert, "build_agent", build_expert)

    def build_dispatcher(shared_model, experts):
        received.append(shared_model)
        assert len(experts) == 3
        return "workflow"

    monkeypatch.setattr(workflow.dispatcher_agent, "build_agent", build_dispatcher)
    assert workflow.build_workflow(model) == "workflow"
    assert len(received) == 4 and all(item is model for item in received)


def test_agent_and_workflow_dependencies():
    root = Path(__file__).parents[1]
    for folder in ("agents", "workflows", "tools"):
        for path in (root / "src/lg_report" / folder).glob("*.py"):
            tree = ast.parse(path.read_text())
            assert ast.get_docstring(tree), path
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("samples"), path
    for path in (root / "samples").glob("*/app.py"):
        source = path.read_text()
        assert "create_deep_agent(" not in source and "create_agent(" not in source, (
            path
        )
    assert not (root / "src/lg_report/platform/scripted_conversation.py").exists()
    assert not list((root / "samples").glob("*/simulation.py"))
    assert not list((root / "samples").glob("*/tools.py"))


def test_domain_lookup_boundaries_and_missing_evidence():
    from lg_report.tools.domain_reference import (
        search_history_reference,
        search_movie_reference,
        search_sports_reference,
    )

    for lookup, question in zip(
        (search_movie_reference, search_sports_reference, search_history_reference),
        ("Spirited Away", "basketball", "Berlin Wall"),
        strict=True,
    ):
        assert "Source:" in lookup.invoke({"query": question})
        assert "No matching" in lookup.invoke({"query": "unlisted subject"})
        assert "No matching" in lookup.invoke({"query": ""})
    assert "No matching" in search_movie_reference.invoke({"query": "basketball"})
