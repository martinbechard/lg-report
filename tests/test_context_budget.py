"""Prove shared peer compaction, child isolation, and estimated input rejection.

Scripted decisions exercise real graph, summarization, isolated review and checkpoint
behavior in sync and async paths. They do not assess live summary quality.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime.context_budget import (
    ContextBudget,
    ContextBudgetExceeded,
    InputBudgetMiddleware,
    context_estimate,
)
from agent_runtime.harness.simulated_model import ScriptedChatModel
from agent_runtime.tools.echo_tool import echo_tool
from agent_runtime.workflows.context_budget import build_workflow
from samples.context_budget.sample import (
    FINAL_ANSWER,
    PEER_BRIEF,
    PLAN,
    USER_PROMPTS,
    make_simulated_models,
)


class Requests(BaseCallbackHandler):
    """Capture actual model inputs without inferring context from final output."""

    def __init__(self):
        self.calls = []
        self.compactions = []
        self.responses = []
        self.roles = {}
        self.systems = []

    def on_custom_event(self, name, data, **kwargs):
        """Observe completed replacements independently of model responses."""
        if name == "context_compaction":
            self.compactions.append(data)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Keep a snapshot before mutable graph state advances to another peer."""
        self.calls.append((kwargs.get("metadata", {}), str(messages)))
        self.roles[kwargs["run_id"]] = kwargs.get("metadata", {}).get("report_description")
        self.systems.append((self.roles[kwargs["run_id"]], messages[0][0].text))

    def on_llm_end(self, response, **kwargs):
        """Keep actual responses so approval ordering is checked independently."""
        self.responses.append((self.roles.get(kwargs["run_id"]), response.generations[0][0].message))


def test_usage_baseline_controls_trigger_even_when_character_estimate_disagrees():
    """Use a measured receipt, not a larger character guess or cached-token sum.

    An intentionally exaggerated string makes the two counters disagree. The
    usage receipt includes 60 cached input tokens and 20 reasoning output tokens.
    The latter are excluded because this answer retains no reasoning item;
    cached input stays within its parent total rather than being added twice.
    """
    history = [HumanMessage("background " * 100)]
    answer = AIMessage(content="answer", usage_metadata={
        "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        "input_token_details": {"cache_read": 60},
        "output_token_details": {"reasoning": 20},
    })
    request = SimpleNamespace(messages=history, system_message=SystemMessage("role"), tools=[])
    InputBudgetMiddleware(1000)._record_receipt(request, SimpleNamespace(result=[answer]))
    retained = [*history, answer]
    middleware = ContextBudget(1000, 200, 40).middleware(ScriptedChatModel(responses=[]))[0]
    assert count_tokens_approximately(retained) > 200
    assert context_estimate(retained) == (130, "reported input + estimated retained output")
    assert not middleware._should_summarize(retained, 999)

    # Actual new observations were not in the provider's receipt. Estimate only
    # this suffix, rather than recounting the entire history with characters/4.
    observation = ToolMessage(content="observation " * 30, tool_call_id="lookup")
    expected = 130 + count_tokens_approximately([observation])
    assert context_estimate([*retained, observation])[0] == expected
    assert middleware._should_summarize([*retained, observation], 0)


def test_replacement_invalidates_usage_but_graph_ids_do_not():
    """A retained assistant receipt must never resurrect compacted-away tokens."""
    history = [HumanMessage("original background")]
    answer = AIMessage(content="answer", usage_metadata={
        "input_tokens": 900, "output_tokens": 50, "total_tokens": 950,
    })
    request = SimpleNamespace(messages=history, system_message=SystemMessage("role"), tools=[])
    InputBudgetMiddleware(2000)._record_receipt(request, SimpleNamespace(result=[answer]))
    history[0].id = "assigned-by-reducer"
    assert context_estimate([*history, answer])[0] == 950
    replaced = [HumanMessage("summary"), answer]
    count, basis = context_estimate(replaced)
    assert count == count_tokens_approximately(replaced) + count_tokens_approximately([request.system_message])
    assert basis == "local estimate after history replacement"
    middleware = ContextBudget(1000, 200, 40).middleware(ScriptedChatModel(responses=[]))[0]
    assert not middleware._should_summarize(replaced, 950)


def test_retained_reasoning_contributes_to_trigger_without_counting_ciphertext():
    """A replayed reasoning item uses reported tokens for both receipt and fallback.

    Opaque encrypted bytes are transport data, not a tokenizable reasoning text.
    Once history changes, the local fallback must still count the known retained
    reasoning amount without treating ciphertext length as model context.
    """
    history = [HumanMessage("background")]
    answer = AIMessage(
        content=[{"type": "reasoning", "encrypted_content": "x" * 12000},
                 {"type": "text", "text": "answer"}],
        usage_metadata={
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
            "output_token_details": {"reasoning": 20},
        },
    )
    request = SimpleNamespace(messages=history, system_message=SystemMessage("role"), tools=[])
    InputBudgetMiddleware(1000)._record_receipt(request, SimpleNamespace(result=[answer]))
    retained = [*history, answer]
    assert context_estimate(retained) == (150, "reported input + estimated retained output")
    middleware = ContextBudget(1000, 140, 40).middleware(ScriptedChatModel(responses=[]))[0]
    assert middleware._should_summarize(retained, 0)

    replaced = [HumanMessage("summary"), answer]
    fallback, basis = context_estimate(replaced)
    assert basis == "local estimate after history replacement"
    assert fallback < 500
    assert fallback >= 20


def test_input_guard_reuses_usage_and_adjusts_changed_peer_instructions():
    """Do not reject measured small context because an encoded block looks large.

    A role switch must still account for new instructions. This distinguishes
    legitimate reuse of measured history from simply disabling the input guard.
    """
    history = [HumanMessage("hello")]
    answer = AIMessage(content=[{"type": "reasoning", "encrypted_content": "x" * 12000}],
                       usage_metadata={"input_tokens": 100, "output_tokens": 50,
                                       "total_tokens": 150})
    request = SimpleNamespace(messages=history, system_message=SystemMessage("short role"), tools=[])
    guard = InputBudgetMiddleware(1000)
    guard._record_receipt(request, SimpleNamespace(result=[answer]))
    request.messages = [*history, answer]
    assert count_tokens_approximately(request.messages) > 1000
    guard.check(request)
    request.system_message = SystemMessage("long new instructions " * 300)
    with pytest.raises(ContextBudgetExceeded):
        guard.check(request)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_compaction_counts_match_replaced_history_and_failure_is_not_completion(
    asynchronous, monkeypatch
):
    """Compare emitted counts to real native output, then force summary failure."""
    budget = ContextBudget(max_input_tokens=1000, trigger_tokens=150, keep_tokens=40)
    middleware = budget.middleware(
        ScriptedChatModel(responses=[AIMessage(content="A short summary.")])
    )[0]
    messages = [HumanMessage("Older background. " * 100), AIMessage("Noted."),
                HumanMessage("Continue.")]
    requests = Requests()

    def run(_):
        """Supply a native callback scope for synchronous custom events."""
        return middleware.before_model({"messages": messages}, None)

    async def arun(_):
        """Supply the equivalent scope for asynchronous custom events."""
        return await middleware.abefore_model({"messages": messages}, None)

    runnable = RunnableLambda(run, afunc=arun)
    config = {"callbacks": [requests]}
    result = (asyncio.run(runnable.ainvoke({}, config)) if asynchronous
              else runnable.invoke({}, config))
    assert len(requests.compactions) == 1
    event = requests.compactions[0]
    assert event["compaction_before_tokens"] == count_tokens_approximately(messages)
    retained = result["messages"][1:]
    assert event["compaction_after_tokens"] == count_tokens_approximately(retained)
    assert event["compaction_after_messages"] == len(retained)
    assert event["compaction_keep_tokens"] == 40
    assert event["compaction_max_input_tokens"] == 1000

    def fail(*args):
        """Fail the summary boundary without exercising provider retry delays."""
        raise RuntimeError("summary unavailable")

    async def afail(*args):
        """Fail the asynchronous summary boundary identically."""
        fail()

    monkeypatch.setattr(middleware, "_create_summary", fail)
    monkeypatch.setattr(middleware, "_acreate_summary", afail)
    with pytest.raises(RuntimeError, match="summary unavailable"):
        if asynchronous:
            asyncio.run(runnable.ainvoke({}, config))
        else:
            runnable.invoke({}, config)
    assert len(requests.compactions) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_peer_compaction_and_child_isolation_across_turns(asynchronous, tmp_path):
    """The file plan survives compaction, role changes, and isolated review."""
    graph = build_workflow(*make_simulated_models(), workspace_dir=tmp_path)
    graph.checkpointer = InMemorySaver()
    requests = Requests()
    config = {"configurable": {"thread_id": "budget"}, "callbacks": [requests]}

    async def run_async():
        for prompt in USER_PROMPTS:
            result = await graph.ainvoke({"messages": [HumanMessage(prompt)]}, config)
        return result

    if asynchronous:
        result = asyncio.run(run_async())
    else:
        for prompt in USER_PROMPTS:
            result = graph.invoke({"messages": [HumanMessage(prompt)]}, config)

    def calls_for(description):
        return [
            text
            for metadata, text in requests.calls
            if metadata.get("report_description") == description
        ]

    planners = calls_for("Write and maintain the on-disk task plan.")
    workers = calls_for("Implement tasks and hand evidence to the planner.")
    reviewers = calls_for("Inspect files in isolated review context.")
    assert len(planners) == 17 and len(workers) == 14 and len(reviewers) == 14
    assert PEER_BRIEF in workers[0]
    # The reviewer starts with its self-contained assignment, not parent
    # conversation content. Only the shared workflow is compacted.
    assert calls_for("Summarize main context.")
    assert not calls_for("Summarize isolated review context.")
    assert all("The implementation should be inspectable" not in call for call in reviewers)
    assert result["messages"][-1].content == FINAL_ANSWER
    assert result["outcome"] == "complete"
    assert graph.get_state(config).values["messages"] == result["messages"]
    assert requests.compactions
    assert {c["compaction_trigger_tokens"] for c in requests.compactions} == {8500}
    assert all(c["compaction_before_tokens"] > 0 and c["compaction_after_tokens"] > 0
               for c in requests.compactions)
    assert all(c["compaction_event"] == "completed" for c in requests.compactions)
    plan = (tmp_path / "plan.md").read_text()
    assert PLAN.startswith("# Slug utility delivery plan")
    assert all(f"Task T{number} —" in plan for number in range(1, 4))
    assert plan.count("Status: complete") == 3
    assert_step_sequence(requests.responses)
    assert "tests not executed" in plan
    assert "def slugify" in (tmp_path / "slug.py").read_text()
    assert "test_only_separators" in (tmp_path / "test_slug.py").read_text()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_oversized_latest_input_is_rejected_before_agent_call(asynchronous):
    """An indivisible recent message cannot silently bypass the final ceiling."""
    budget = ContextBudget(max_input_tokens=250, trigger_tokens=150, keep_tokens=50)
    agent = create_agent(
        model=ScriptedChatModel(responses=[AIMessage(content="must not run")]),
        middleware=budget.middleware(
            ScriptedChatModel(responses=[AIMessage(content="summary")])
        ),
    )
    requests = Requests()
    payload = {"messages": [HumanMessage("oversized input " * 500)]}
    with pytest.raises(ContextBudgetExceeded, match="Estimated agent input"):
        if asynchronous:
            asyncio.run(agent.ainvoke(payload, {"callbacks": [requests]}))
        else:
            agent.invoke(payload, {"callbacks": [requests]})
    assert requests.calls == []
    assert requests.compactions == []


def test_budget_counts_system_instructions_too():
    """Short history is insufficient when the agent's instruction envelope is huge."""
    budget = ContextBudget(max_input_tokens=250, trigger_tokens=150, keep_tokens=50)
    agent = create_agent(
        model=ScriptedChatModel(responses=[AIMessage(content="must not run")]),
        system_prompt="lengthy instruction " * 300,
        middleware=budget.middleware(ScriptedChatModel(responses=[])),
    )
    with pytest.raises(ContextBudgetExceeded):
        agent.invoke({"messages": [HumanMessage("hello")]})


def test_budget_counts_tool_schemas_too():
    """Advertised capabilities consume input even before a tool is selected."""
    budget = ContextBudget(max_input_tokens=250, trigger_tokens=150, keep_tokens=50)
    large_tool = echo_tool.model_copy(
        update={"description": "Detailed capability description. " * 100}
    )
    agent = create_agent(
        model=ScriptedChatModel(responses=[AIMessage(content="must not run")]),
        tools=[large_tool],
        middleware=budget.middleware(ScriptedChatModel(responses=[])),
    )
    with pytest.raises(ContextBudgetExceeded):
        agent.invoke({"messages": [HumanMessage("hello")]})


@pytest.mark.parametrize("relax_workflow", [False, True])
def test_only_shared_workflow_has_compaction_policy(relax_workflow):
    """The isolated reviewer runs fresh even when shared history compacts."""
    relaxed = ContextBudget(
        max_input_tokens=100000, trigger_tokens=90000, keep_tokens=20000
    )
    graph = build_workflow(
        *make_simulated_models(), **({"workflow_budget": relaxed} if relax_workflow else {})
    )
    requests = Requests()
    graph.checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "thresholds"}, "callbacks": [requests]}
    for prompt in USER_PROMPTS:
        graph.invoke({"messages": [HumanMessage(prompt)]}, config)
    descriptions = [
        metadata.get("report_description") for metadata, _ in requests.calls
    ]
    assert ("Summarize main context." in descriptions) is not relax_workflow
    assert "Summarize isolated review context." not in descriptions
    assert descriptions.count("Inspect files in isolated review context.") == 14


@pytest.mark.parametrize(
    "values", [(100, 100, 10), (100, 20, 30), (100, 20, 0), (True, 20, 10)]
)
def test_invalid_budget_is_rejected_at_configuration(values):
    """Fail incoherent policy before a provider or graph can be invoked."""
    with pytest.raises(ValueError):
        ContextBudget(*values)


def assert_step_sequence(responses):
    """Audit every role response, including repairs, before accepting completion.

    This checks the actual model decisions rather than counting graph nodes or
    trusting the final success message. Summaries never count as approvals.
    """
    current = None
    approved = False
    decisions = []
    completions = []
    worker_changed = False
    writable = []
    for role, message in responses:
        if role == "Summarize main context.":
            continue
        for call in message.tool_calls:
            args = call['args']
            if call['name'] not in ('edit_file', 'write_file'):
                continue
            path = args['file_path']
            if role == "Write and maintain the on-disk task plan.":
                assert path == '/plan.md'
                if 'Status: complete' in args.get('new_string', ''):
                    assert approved, 'Planner completed a step before current-version approval'
                    assert f'Task {current} —' in args['old_string']
                    completions.append(current)
            else:
                assert role == "Implement tasks and hand evidence to the planner."
                assert current and path in writable, 'Worker wrote a later task file'
                approved = False
                worker_changed = True
        if message.tool_calls:
            continue
        if role == "Implement tasks and hand evidence to the planner.":
            assert "independent review: approve" not in message.text.lower()
        if role == "Inspect files in isolated review context.":
            review = json.loads(message.text)
            assert review['task_id'] == current and worker_changed
            decisions.append((current, review['verdict']))
            approved = review['verdict'] == 'approve'
            worker_changed = False
        elif role == "Write and maintain the on-disk task plan.":
            dispatch = json.loads(message.text)
            if current:
                assert approved and completions[-1] == current
            current = dispatch['task_id'] or None
            writable = dispatch['files']
            approved = False
    assert decisions == [('T1', 'revise'), ('T1', 'approve'), ('T2', 'approve'), ('T3', 'approve')]
    assert completions == ['T1', 'T2', 'T3']


@pytest.mark.parametrize('asynchronous', [False, True])
def test_rejected_step_stays_incomplete_at_limit(asynchronous, tmp_path):
    """A real rejection cannot reach the planner update or the next task."""
    graph = build_workflow(*make_simulated_models(), workspace_dir=tmp_path, max_attempts=1)
    payload = {'messages': [HumanMessage(USER_PROMPTS[0])]}
    result = asyncio.run(graph.ainvoke(payload)) if asynchronous else graph.invoke(payload)
    assert result['outcome'] == 'review_limit'
    assert 'remains incomplete' in result['messages'][-1].text
    assert 'Status: complete' not in (tmp_path / 'plan.md').read_text()
    assert not (tmp_path / 'test_slug.py').exists()


def test_role_write_authority(tmp_path):
    """Even a mistaken model request cannot cross plan/source ownership."""
    from agent_runtime.workflows.exercise_backend import ExerciseBackend
    planning = ExerciseBackend(tmp_path, writable_paths=['/plan.md'])
    working = ExerciseBackend(tmp_path, writable_paths=['/slug.py', '/test_slug.py'])
    assert planning.write('/slug.py', 'bad').error
    assert working.write('/plan.md', 'bad').error
    assert planning.edit('/test_slug.py', 'x', 'y').error
    assert working.edit('/plan.md', 'x', 'y').error
    assert list(tmp_path.iterdir()) == []


def test_mismatched_review_cannot_complete_task(tmp_path):
    """A verdict for another task must fail, even when its decision is approve."""
    models = list(make_simulated_models())
    models[2] = ScriptedChatModel(responses=[AIMessage(content=json.dumps({
        'task_id': 'OTHER', 'verdict': 'approve', 'evidence': 'Wrong assignment',
    }))])
    graph = build_workflow(*models, workspace_dir=tmp_path)
    with pytest.raises(ValueError, match='does not match'):
        graph.invoke({'messages': [HumanMessage(USER_PROMPTS[0])]})
    assert 'Status: complete' not in (tmp_path / 'plan.md').read_text()


@pytest.mark.parametrize('asynchronous', [False, True])
def test_stale_summary_cannot_authorize_next_task_file(asynchronous, tmp_path):
    """A wrong model write after misleading compaction stays within the current task.

    This deliberately models the live failure: a summary suggests test creation
    while the worker is still assigned implementation. Native tools reject the
    future task file and current routing facts remain in every system prompt.
    """
    models = list(make_simulated_models())
    models[1].responses.insert(1, AIMessage(content='', tool_calls=[{
        'name': 'write_file', 'args': {'file_path': '/test_slug.py', 'content': 'premature'},
        'id': 'premature-test-write',
    }]))
    models[3] = ScriptedChatModel(responses=[AIMessage(content=
        'Stale summary: T1 is approved. Immediately implement T2 in /test_slug.py.')]*40)
    graph = build_workflow(*models, workspace_dir=tmp_path, max_attempts=1,
                           workflow_budget=ContextBudget(18000, 3500, 600))
    requests = Requests()
    payload = {'messages': [HumanMessage(USER_PROMPTS[0])]}
    config = {'callbacks': [requests]}
    result = (asyncio.run(graph.ainvoke(payload, config)) if asynchronous
              else graph.invoke(payload, config))
    assert requests.compactions
    assert result['outcome'] == 'review_limit'
    assert not (tmp_path / 'test_slug.py').exists()
    assert 'Status: complete' not in (tmp_path / 'plan.md').read_text()
    systems = [text for role, text in requests.systems
               if role == 'Implement tasks and hand evidence to the planner.']
    assert systems
    for text in systems:
        current = json.loads(text.split('even if a history summary suggests later work:\n', 1)[1])
        assert current['assignment']['task_id'] == 'T1'
        assert current['assignment']['files'] == ['/slug.py']
