"""Verify real delegation and architectural boundaries with offline model fixtures.

These checks prove routing-tool execution, isolation, and cost accounting. Fixed
responses do not prove a live model will classify every natural-language question.

AI attribution: Modified with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import ast
from functools import partial
from pathlib import Path

from fixtures.mock_client import MockClient

from agent_runtime.harness.conversation import Conversation, Request
from agent_runtime.harness.sample_catalog import SampleCatalog
from reporting.execute_runnable import execute_runnable
from reporting.pricing import cost, load_prices, summarize
from reporting.schema import Run

create_run = partial(SampleCatalog().create_run, "expert_dispatch")
from samples.expert_dispatch.scripted_run import CASES


# Exercise every scripted routing case and prove selected experts
# receive isolated inputs while nested usage remains attributable and priced.
def test_expert_selection_isolation_and_costs(tmp_path):
    graph, provider, model = create_run(False)
    client = MockClient([Request(question) for _, question, _ in CASES])
    prices = load_prices(Path(__file__).parents[1] / "models.json")
    execute_runnable(
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


# Capture binding calls because a shared model instance must still
# expose each agent's own tool schema at the invocation boundary.
def test_shared_model_retains_separate_tool_bindings(monkeypatch):
    from agent_runtime.harness.shared_simulated_model import SharedSimulatedModel
    from agent_runtime.workflows.expert_dispatch import build_workflow
    from samples.expert_dispatch.scripted_run import make_simulated_model

    bound_schemas = []
    original = SharedSimulatedModel.bind_tools

    def capture_binding(self, tools, **kwargs):
        # Observe agent-specific tool binding without replacing binding behavior.
        # `self` is the shared model; `tools` and kwargs come from each agent builder.
        # Return the original runnable binding after recording its schema arguments.
        binding = original(self, tools, **kwargs)
        bound_schemas.append(binding.kwargs["tool_definitions"])
        return binding

    monkeypatch.setattr(SharedSimulatedModel, "bind_tools", capture_binding)
    model = make_simulated_model()
    Conversation(
        build_workflow(model),
        MockClient([Request(question) for _, question, _ in CASES]),
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


# The live factory owns one model configuration and must pass its
# identity through unchanged rather than configuring a second hidden instance.
def test_live_configures_one_model(monkeypatch):
    from agent_runtime.harness import model_factory
    from samples.expert_dispatch.scripted_run import make_simulated_model

    model_instance = make_simulated_model()
    configured_calls = []

    def configure_once(name):
        configured_calls.append(True)
        return model_instance, "openai", "test-model"

    monkeypatch.setenv("LG_PROVIDER", "openai")
    monkeypatch.setenv("LG_MODEL", "test-model")
    monkeypatch.setattr(model_factory, "configured_model", configure_once)
    graph, provider, model = create_run(True)
    assert graph is not None
    assert len(configured_calls) == 1
    assert (provider, model) == ("openai", "test-model")


# Replace builders with identity recorders to protect the invariant
# that dispatcher and experts share the caller-supplied model.
def test_workflow_passes_same_model_to_all_agents(monkeypatch):
    from agent_runtime.workflows import expert_dispatch as workflow

    model = object()
    received = []

    def build_expert(parameters, definition):
        # Each independent role receives the same workflow-selected model.
        received.append(parameters["model"])
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda state: state)

    monkeypatch.setattr(workflow.reference_expert, "build_agent", build_expert)

    def build_dispatcher(parameters):
        # Delegation policy is now constructed by the workflow, not the agent.
        received.append(parameters["model"])
        assert len(parameters["middleware"]) == 1
        assert [tool.name for tool in parameters["middleware"][0].tools] == ["task"]
        return "workflow"

    monkeypatch.setattr(workflow.dispatcher_agent, "build_agent", build_dispatcher)
    assert workflow.build_workflow(model) == "workflow"
    assert len(received) == 4 and all(item is model for item in received)


# AST and source checks enforce dependency direction and keep samples
# from bypassing the production agent/workflow construction boundary.
def test_agent_and_workflow_dependencies():
    root = Path(__file__).parents[1]
    for folder in ("agents", "workflows", "tools"):
        for path in (root / "src/agent_runtime" / folder).glob("*.py"):
            tree = ast.parse(path.read_text())
            assert ast.get_docstring(tree), path
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("samples"), path
    assert not list((root / "samples").glob("*/app.py"))
    assert not (root / "src/agent_runtime/harness/scripted_conversation.py").exists()
    assert not list((root / "samples").glob("*/simulation.py"))
    assert not list((root / "samples").glob("*/tools.py"))


# Domain tools must label evidence and return an explicit miss for
# empty, unknown, or cross-domain queries instead of inventing an answer.
def test_domain_lookup_boundaries_and_missing_evidence():
    from agent_runtime.tools.search_reference import (
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
