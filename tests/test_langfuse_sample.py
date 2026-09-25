"""Verify official Langfuse callback behavior without a hosted tracing service.

Keep the real SDK and graph execution, but replace the network exporter with an
in-memory exporter. Assertions cover trace ancestry, conversation history, usage,
and failure handling; they do not prove remote ingestion or browser access.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from uuid import uuid4

import pytest
from fixtures.mock_client import MockClient
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent_runtime.harness import app, model_factory
from agent_runtime.harness import langfuse_runtime as runtime
from agent_runtime.harness.conversation import Request
from agent_runtime.harness.sample_catalog import SampleCatalog
from samples.simple_chat.sample import CONVERSATION

# Expected values are test projections of the authored conversation.
USER_PROMPTS = [entry["content"] for entry in CONVERSATION if entry["role"] == "client"]


def trace_conversation(
    graph, client, callback, *, chat_client, simulated, trace_name, public_trace=False
):
    """Exercise tracing hooks with the real loop independently of CLI setup.

    Production local-report coverage below runs through execute_conversation. These
    SDK tests isolate observation ancestry, publication, and interruption handling.
    """
    from langsmith import tracing_context

    from agent_runtime.harness.conversation import Conversation

    with (
        tracing_context(enabled=False),
        runtime.conversation_trace(
            client,
            trace_name=trace_name,
            simulated=simulated,
            public_trace=public_trace,
        ) as trace,
    ):
        result = Conversation(graph, chat_client, turn_scope=trace.turn_scope).invoke(
            {}, {"callbacks": [callback]}
        )
        trace.complete(result)
    return trace.trace_id, result.get("messages", []) if result else []


@pytest.fixture
def capture():
    """Keep the production SDK/callback; replace only its network exporter."""
    # A distinct key isolates SDK client lookup across tests. Yield returns
    # the SDK client, its callback, and the local exporter to the test; pytest
    # re-enters the fixture after the test so finally shuts down the client.
    key = f"pk-lf-test-{uuid4().hex}"
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=key,
        secret_key="test-only",
        base_url="http://localhost:1",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        sample_rate=1.0,
        tracing_enabled=True,
    )
    try:
        yield client, CallbackHandler(public_key=key), exporter
    finally:
        client.shutdown()


# Two turns should share one trace while retaining per-turn usage and
# conversation ancestry in the in-memory exporter.
def test_two_turns_share_trace_and_preserve_usage(capture):
    """Check topology, growing context, usage partitions, and model annotations."""
    client, callback, exporter = capture
    trace_id, history = trace_conversation(
        SampleCatalog().create_run("simple_chat", False)[0],
        client,
        callback,
        simulated=True,
        chat_client=MockClient([Request(p) for p in USER_PROMPTS]),
        trace_name="simple-chat-langfuse",
    )
    client.flush()
    spans = exporter.get_finished_spans()
    assert len({s.context.trace_id for s in spans}) == 1
    assert f"{spans[0].context.trace_id:032x}" == trace_id
    root = next(s for s in spans if s.name == "simple-chat-langfuse")
    assert root.attributes.get("langfuse.trace.public", False) is False
    turns = [s for s in spans if s.name in {"Turn 1", "Turn 2"}]
    assert len(turns) == 2
    assert all(s.parent.span_id == root.context.span_id for s in turns)
    generations = sorted(
        (
            s
            for s in spans
            if s.attributes.get("langfuse.observation.type") == "generation"
        ),
        key=lambda s: s.start_time,
    )
    assert len(generations) == 2
    assert [m.type for m in history] == ["human", "ai", "human", "ai"]
    by_id = {s.context.span_id: s for s in spans}
    for index, span in enumerate(generations):
        # Walk actual parents rather than assuming how many middleware nodes exist.
        ancestor = span
        while ancestor.name not in {"Turn 1", "Turn 2"}:
            ancestor = by_id[ancestor.parent.span_id]
        assert ancestor.name == f"Turn {index + 1}"
        usage = json.loads(span.attributes["langfuse.observation.usage_details"])
        expected = history[index * 2 + 1].usage_metadata
        cached = expected["input_token_details"]["cache_read"]
        assert usage["input"] + usage["input_cache_read"] == expected["input_tokens"]
        assert usage["input_cache_read"] == cached
        assert usage["output"] == expected["output_tokens"]
        assert span.attributes["langfuse.observation.metadata.report_effort"] == "light"
        assert span.attributes["langfuse.observation.metadata.report_turn"] == index + 1
    second_input = json.loads(generations[1].attributes["langfuse.observation.input"])
    assert [m["role"] for m in second_input] == ["system", "user", "assistant", "user"]
    assert second_input[2]["content"] == history[1].content
    assert (
        history[3].usage_metadata["input_tokens"]
        > history[1].usage_metadata["input_tokens"]
    )


# Graph failure must close exported spans with error status so a
# hosted exporter would not receive an open or falsely successful trace.
def test_failed_graph_closes_error_spans(capture):
    """Partial failures must remain observable and must not start a second turn."""
    client, callback, exporter = capture

    def fail(_):
        # Force graph failure after tracing has started to exercise span cleanup.
        # RunnableLambda supplies the graph input as `_`; its content is irrelevant.
        raise RuntimeError("deliberate test failure")

    builder = StateGraph(MessagesState)
    builder.add_node("fail", fail)
    builder.add_edge(START, "fail")
    builder.add_edge("fail", END)
    with pytest.raises(RuntimeError, match="deliberate test failure"):
        trace_conversation(
            builder.compile(),
            client,
            callback,
            simulated=True,
            chat_client=MockClient([Request(p) for p in USER_PROMPTS]),
            trace_name="simple-chat-langfuse",
        )
    client.flush()
    spans = exporter.get_finished_spans()
    assert not any(s.name == "Turn 2" for s in spans)
    for name in ("Turn 1", "simple-chat-langfuse"):
        span = next(s for s in spans if s.name == name)
        assert span.end_time is not None
        assert span.status.status_code == StatusCode.ERROR


# Missing Langfuse configuration is a startup error and must prevent
# client construction, avoiding a misleading partially initialized runtime.
def test_missing_configuration_fails_before_client_creation(
    monkeypatch, tmp_path, capsys
):
    """A missing project never silently disables tracing or runs another backend."""
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "app",
            "--sample",
            "simple_chat_langfuse",
            "--env-file",
            str(tmp_path / "absent.env"),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        app.main()
    assert failure.value.code == 2
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err


# Authentication failure must stop before model selection, preserving
# the documented initialization order and avoiding provider work.
def test_failed_authentication_stops_before_model_selection(monkeypatch):
    """The CLI checks project access before constructing any live provider model."""
    for name, value in {
        "LANGFUSE_BASE_URL": "http://localhost:1",
        "LANGFUSE_PUBLIC_KEY": "test-public",
        "LANGFUSE_SECRET_KEY": "test-secret",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "sys.argv", ["app", "--sample", "simple_chat_langfuse", "--live"]
    )
    shutdown = []

    class UnauthenticatedClient:
        def auth_check(self):
            # Simulate denied project access before any provider work can begin.
            # Return the SDK authentication check boolean without contacting Langfuse.
            return False

        def shutdown(self):
            # Record cleanup after authentication failure so a failed startup cannot
            # silently leave the constructed client running.
            shutdown.append(True)

    monkeypatch.setattr(runtime, "Langfuse", lambda **kwargs: UnauthenticatedClient())
    monkeypatch.setattr(
        model_factory,
        "configured_model",
        lambda name: pytest.fail("Model created before auth"),
    )
    with pytest.raises(SystemExit) as failure:
        app.main()
    assert failure.value.code == 2
    assert shutdown == [True]


# Delegated child spans must remain under the task span in exported
# trace ancestry so ownership and cost reports agree.
def test_langfuse_delegation_keeps_child_under_task(capture):
    """The actual task tool must propagate one callback into the child's graph."""
    from agent_runtime.harness.sample_catalog import SampleCatalog
    from samples.subagent_chat.sample import CONVERSATION

    # Expected values are test projections of the authored conversation.
    DELEGATED_TASK = CONVERSATION[1]["tool_calls"][0]["args"]["description"]
    SPECIALIST_SUMMARY = CONVERSATION[4]["content"]
    USER_PROMPTS = [
        entry["content"] for entry in CONVERSATION if entry["role"] == "client"
    ]

    client, callback, exporter = capture
    trace_id, history = trace_conversation(
        SampleCatalog().create_run("subagent_chat", False)[0],
        client,
        callback,
        simulated=True,
        chat_client=MockClient([Request(p) for p in USER_PROMPTS]),
        trace_name="subagent-chat-langfuse",
    )
    client.flush()
    spans = exporter.get_finished_spans()
    assert {f"{s.context.trace_id:032x}" for s in spans} == {trace_id}
    models = sorted(
        [
            s
            for s in spans
            if s.attributes.get("langfuse.observation.type") == "generation"
        ],
        key=lambda s: s.start_time,
    )
    assert len(models) == 4
    task = next(s for s in spans if s.name == "task")
    by_id = {s.context.span_id: s for s in spans}
    for model in models[1:3]:
        ancestor = model
        while ancestor.context.span_id != task.context.span_id:
            ancestor = by_id[ancestor.parent.span_id]
    child_input = json.loads(models[1].attributes["langfuse.observation.input"])
    assert child_input[-1]["content"] == DELEGATED_TASK
    assert USER_PROMPTS[0] not in str(child_input)
    parent_input = json.loads(models[3].attributes["langfuse.observation.input"])
    assert parent_input[-1]["content"] == SPECIALIST_SUMMARY
    assert "specialist-lookup-1" not in str(parent_input)
    usages = [
        json.loads(s.attributes["langfuse.observation.usage_details"]) for s in models
    ]
    assert usages[0]["input_cache_read"] == usages[1]["input_cache_read"] == 0
    assert usages[2]["input_cache_read"] > 0 and usages[3]["input_cache_read"] > 0
    assert (
        len(history) == 4
    )  # Only the parent's prompt, task request/result, and answer.


# Console file attachments must produce the same conversation content
# with and without tracing; instrumentation cannot alter user-visible inputs.
def test_console_files_match_untraced_conversation(capture, tmp_path):
    """Changing recording must not change messages or charge a model call twice."""
    from agent_runtime.harness.console_client import ConsoleClient
    from agent_runtime.harness.conversation import Conversation

    path = tmp_path / "note.txt"
    path.write_text("The limit is 42.")
    entries = iter([f"/attach {path}", "Explain this", "And why?", "/quit"])
    displayed = []
    console = ConsoleClient(read=lambda _: next(entries), write=displayed.append)
    client, callback, exporter = capture
    _, history = trace_conversation(
        SampleCatalog().create_run("simple_chat_langfuse", False, tracing=False)[0],
        client,
        callback,
        simulated=True,
        chat_client=console,
        trace_name="console-test",
        public_trace=True,
    )
    baseline = Conversation(
        SampleCatalog().create_run("simple_chat_langfuse", False, tracing=False)[0],
        MockClient(
            [
                Request("Explain this\n\nAttached file: note.txt\nThe limit is 42."),
                Request("And why?"),
            ]
        ),
    ).invoke({}, {})
    assert [(m.type, m.content) for m in history] == [
        (m.type, m.content) for m in baseline["messages"]
    ]
    client.flush()
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "console-test")
    assert root.attributes["langfuse.trace.public"] is True
    assert "The limit is 42." in root.attributes["langfuse.observation.input"]
    assert (
        sum(
            s.attributes.get("langfuse.observation.type") == "generation" for s in spans
        )
        == 2
    )
    assert sum(line.startswith("Assistant:") for line in displayed) == 2


# Empty and interrupted sessions exercise terminal lifecycle edges;
# both must close cleanly and retain truthful trace statuses.
def test_empty_and_interrupted_sessions(capture):
    """Neither a user exit nor an approval pause should consume another prompt."""
    client, callback, exporter = capture
    _, history = trace_conversation(
        SampleCatalog().create_run("simple_chat_langfuse", False, tracing=False)[0],
        client,
        callback,
        simulated=True,
        chat_client=MockClient([]),
        trace_name="empty",
    )
    assert history == []

    def ask(state):
        answer = interrupt({"kind": "question", "question": "Continue?"})
        return {"messages": [AIMessage(content=answer)]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    user = MockClient([Request("first")], answer=lambda _: "Confirmed")
    _, history = trace_conversation(
        builder.compile(),
        client,
        callback,
        simulated=True,
        chat_client=user,
        trace_name="pause",
    )
    assert history[-1].content == "Confirmed"
    client.flush()
    spans = exporter.get_finished_spans()
    for name, expected in [("empty", "incomplete"), ("pause", "ok")]:
        root = next(s for s in spans if s.name == name)
        assert (
            root.attributes["langfuse.observation.metadata.session_status"] == expected
        )
    assert not any(s.name == "Turn 2" for s in spans)


@pytest.mark.parametrize(
    "sample_id,calls", [("simple_chat_langfuse", 2), ("subagent_chat_langfuse", 4)]
)
@pytest.mark.parametrize("capture_content", [False, True])
def test_langfuse_launch_always_records_locally(
    capture, monkeypatch, tmp_path, sample_id, calls, capture_content
):
    """One execution must reach both exporters and honor the local capture setting."""
    from pathlib import Path

    from agent_runtime.harness.configure_sample_script import configure_sample_script
    from agent_runtime.harness.console_client import ConsoleClient
    from agent_runtime.harness.execute_conversation import execute_conversation
    from agent_runtime.harness.settings import Settings
    from reporting.pricing import load_prices

    sdk, callback, exporter = capture
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "test-only")
    monkeypatch.setattr(runtime, "Langfuse", lambda **kwargs: sdk)
    monkeypatch.setattr(runtime, "CallbackHandler", lambda **kwargs: callback)
    monkeypatch.setattr(sdk, "auth_check", lambda: True)
    monkeypatch.setattr(
        sdk, "get_trace_url", lambda **kwargs: "http://localhost/trace/test"
    )
    catalog = SampleCatalog()
    client = ConsoleClient(write=lambda _: None)
    configure_sample_script(client, catalog, sample_id, script_answers=True)
    settings = Settings(
        False,
        tmp_path,
        load_prices(Path(__file__).parent / "fixtures/accounting_prices.json"),
        capture_content,
        True,
    )
    execute_conversation(
        catalog=catalog,
        sample_id=sample_id,
        settings=settings,
        options={},
        client=client,
    )
    assert all(
        (tmp_path / name).is_file()
        for name in ("run.json", "spans.jsonl", "prices.json", "report.html")
    )
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["status"] == "ok"
    assert sum(step["kind"] == "model" for step in run["steps"]) == calls
    generations = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("langfuse.observation.type") == "generation"
    ]
    assert len(generations) == calls
    if not capture_content:
        assert run.get("output") is None
    else:
        assert run.get("output")


def test_langfuse_failure_still_writes_local_evidence(capture, monkeypatch, tmp_path):
    """A failed graph saves its partial local report before propagating the error."""
    from agent_runtime.harness.execute_conversation import execute_conversation
    from agent_runtime.harness.settings import Settings
    from reporting.pricing import Prices

    sdk, callback, exporter = capture
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "test-only")
    monkeypatch.setattr(runtime, "Langfuse", lambda **kwargs: sdk)
    monkeypatch.setattr(runtime, "CallbackHandler", lambda **kwargs: callback)
    monkeypatch.setattr(sdk, "auth_check", lambda: True)
    monkeypatch.setattr(
        sdk, "get_trace_url", lambda **kwargs: "http://localhost/trace/test"
    )

    def fail(_):
        raise RuntimeError("deliberate dual-recording failure")

    builder = StateGraph(MessagesState)
    builder.add_node("failure", fail)
    builder.add_edge(START, "failure")
    builder.add_edge("failure", END)
    catalog = SampleCatalog()
    monkeypatch.setattr(
        catalog,
        "create_run",
        lambda *a, **kw: (builder.compile(), "demo", "scripted-chat"),
    )
    settings = Settings(
        False,
        tmp_path,
        Prices(as_of="2026-09-22", note="Test fixture", models={}),
        True,
        True,
    )
    with pytest.raises(RuntimeError, match="deliberate dual-recording failure"):
        execute_conversation(
            catalog=catalog,
            sample_id="simple_chat_langfuse",
            settings=settings,
            options={},
            client=MockClient([Request("fail")]),
        )
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "error"
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "spans.jsonl").exists()
    assert any(
        span.status.status_code == StatusCode.ERROR
        for span in exporter.get_finished_spans()
    )


@pytest.mark.parametrize("tracing", ["local", "langfuse"])
def test_structured_status_and_cleanup_share_launch_path(
    capture, monkeypatch, tmp_path, capsys, tracing
):
    """Tracing must not bypass structured status output or workflow resource cleanup."""
    from dataclasses import replace
    from types import SimpleNamespace

    from agent_runtime.harness.configure_sample_script import configure_sample_script
    from agent_runtime.harness.console_client import ConsoleClient
    from agent_runtime.harness.execute_conversation import execute_conversation
    from agent_runtime.harness.settings import Settings
    from reporting.pricing import Prices

    sdk, callback, _ = capture
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "test-only")
    monkeypatch.setattr(runtime, "Langfuse", lambda **kwargs: sdk)
    monkeypatch.setattr(runtime, "CallbackHandler", lambda **kwargs: callback)
    monkeypatch.setattr(sdk, "auth_check", lambda: True)
    monkeypatch.setattr(
        sdk, "get_trace_url", lambda **kwargs: "http://localhost/trace/test"
    )
    catalog = SampleCatalog()
    sample_id = "quote_request"
    catalog.samples[sample_id] = replace(catalog.get(sample_id), tracing=tracing)
    build = catalog.create_run
    closed = []

    def create(*args, **kwargs):
        graph, provider, model = build(*args, **kwargs)
        graph.workspace = SimpleNamespace(cleanup=lambda: closed.append(True))
        return graph, provider, model

    monkeypatch.setattr(catalog, "create_run", create)
    client = ConsoleClient(write=lambda _: None)
    configure_sample_script(client, catalog, sample_id, script_answers=True)
    execute_conversation(
        catalog=catalog,
        sample_id=sample_id,
        settings=Settings(
            False,
            tmp_path,
            Prices(as_of="2026-09-22", note="Test fixture", models={}),
            False,
            True,
        ),
        options={},
        client=client,
    )
    assert client.last_status is not None
    assert json.dumps({"status": client.last_status}) in capsys.readouterr().out
    assert closed == [True]
    assert (tmp_path / "report.html").is_file()
