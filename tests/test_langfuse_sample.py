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
from langchain_core.runnables import RunnableLambda
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lg_report.platform import langfuse_runtime as runtime
from lg_report.platform.conversation import Request
from lg_report.platform.static_client import StaticClient
from samples.simple_chat.test_case import make_simulated_model
from samples.simple_chat_langfuse import app


@pytest.fixture
def capture():
    """Keep the production SDK/callback; replace only its network exporter."""
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


def test_two_turns_share_trace_and_preserve_usage(capture):
    """Check topology, growing context, usage partitions, and model annotations."""
    client, callback, exporter = capture
    trace_id, history = runtime.run_conversation(
        app.build_workflow(make_simulated_model()),
        client,
        callback,
        simulated=True,
        chat_client=StaticClient([Request(p) for p in app.USER_PROMPTS]),
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


def test_failed_graph_closes_error_spans(capture):
    """Partial failures must remain observable and must not start a second turn."""
    client, callback, exporter = capture

    def fail(_):
        raise RuntimeError("deliberate test failure")

    with pytest.raises(RuntimeError, match="deliberate test failure"):
        runtime.run_conversation(
            RunnableLambda(fail),
            client,
            callback,
            simulated=True,
            chat_client=StaticClient([Request(p) for p in app.USER_PROMPTS]),
            trace_name="simple-chat-langfuse",
        )
    client.flush()
    spans = exporter.get_finished_spans()
    assert not any(s.name == "Turn 2" for s in spans)
    for name in ("Turn 1", "simple-chat-langfuse"):
        span = next(s for s in spans if s.name == name)
        assert span.end_time is not None
        assert span.status.status_code == StatusCode.ERROR


def test_missing_configuration_fails_before_client_creation(
    monkeypatch, tmp_path, capsys
):
    """A missing project never silently disables tracing or runs another backend."""
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.argv", ["app", "--env-file", str(tmp_path / "absent.env")])
    with pytest.raises(SystemExit) as failure:
        app.main()
    assert failure.value.code == 2
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err


def test_failed_authentication_stops_before_model_selection(monkeypatch):
    """The CLI checks project access before constructing any live provider model."""
    for name, value in {
        "LANGFUSE_BASE_URL": "http://localhost:1",
        "LANGFUSE_PUBLIC_KEY": "test-public",
        "LANGFUSE_SECRET_KEY": "test-secret",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("sys.argv", ["app", "--live"])
    shutdown = []

    class UnauthenticatedClient:
        def auth_check(self):
            return False

        def shutdown(self):
            shutdown.append(True)

    monkeypatch.setattr(runtime, "Langfuse", lambda **kwargs: UnauthenticatedClient())
    monkeypatch.setattr(
        app, "configured_model", lambda: pytest.fail("Model created before auth")
    )
    with pytest.raises(SystemExit) as failure:
        app.main()
    assert failure.value.code == 1
    assert shutdown == [True]


def test_langfuse_delegation_keeps_child_under_task(capture):
    """The actual task tool must propagate one callback into the child's graph."""
    from samples.subagent_chat.app import USER_PROMPTS, create_graph
    from samples.subagent_chat.test_case import DELEGATED_TASK, SPECIALIST_SUMMARY

    client, callback, exporter = capture
    trace_id, history = runtime.run_conversation(
        create_graph(False),
        client,
        callback,
        simulated=True,
        chat_client=StaticClient([Request(p) for p in USER_PROMPTS]),
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


def test_console_files_match_untraced_conversation(capture, tmp_path):
    """Changing recording must not change messages or charge a model call twice."""
    from lg_report.platform.console_client import ConsoleClient
    from lg_report.platform.conversation import Conversation

    path = tmp_path / "note.txt"
    path.write_text("The limit is 42.")
    entries = iter([f"/attach {path}", "Explain this", "And why?", "/quit"])
    displayed = []
    console = ConsoleClient(read=lambda _: next(entries), write=displayed.append)
    client, callback, exporter = capture
    _, history = runtime.run_conversation(
        app.create_graph(False),
        client,
        callback,
        simulated=True,
        chat_client=console,
        trace_name="console-test",
        public_trace=True,
    )
    baseline = Conversation(
        app.create_graph(False),
        StaticClient(
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


def test_empty_and_interrupted_sessions(capture):
    """Neither a user exit nor an approval pause should consume another prompt."""
    client, callback, exporter = capture
    _, history = runtime.run_conversation(
        app.create_graph(False),
        client,
        callback,
        simulated=True,
        chat_client=StaticClient([]),
        trace_name="empty",
    )
    assert history == []
    user = StaticClient([Request("first"), Request("do not send")])
    _, history = runtime.run_conversation(
        RunnableLambda(lambda _: {"messages": [], "__interrupt__": ["approval"]}),
        client,
        callback,
        simulated=True,
        chat_client=user,
        trace_name="pause",
    )
    assert history == []
    assert user.receive().prompt == "do not send"
    client.flush()
    spans = exporter.get_finished_spans()
    for name, expected in [("empty", "incomplete"), ("pause", "interrupted")]:
        root = next(s for s in spans if s.name == name)
        assert (
            root.attributes["langfuse.observation.metadata.session_status"] == expected
        )
    assert not any(s.name == "Turn 2" for s in spans)
