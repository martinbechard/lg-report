"""Prove console/SSE sample parity through the official AG-UI adapter.

Tests execute real sample graphs with scripted models and fixture retrieval.
Native streaming/cancellation is verified separately with a gated chat model.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ag_ui_langgraph")

from ag_ui.core import RunAgentInput, UserMessage
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_runtime.harness.console_client import ConsoleClient
from agent_runtime.harness.conversation import Conversation
from agent_runtime.harness.execution import open_graph
from agent_runtime.harness.sample_catalog import SampleCatalog
from agent_runtime.harness.web_conversation import WebConversation

catalog = SampleCatalog()
SAMPLES = catalog.samples
create_sample = catalog.create_run
sample_info = catalog.info
from agent_runtime.web.server import create_app


@pytest.fixture
def retrieval_fixture(monkeypatch):
    """Exercise real retrieval and MCP tools without downloading a corpus."""
    from deepagents import create_deep_agent
    from langchain.mcp import MCPAdapter
    from test_mcp_rag_sample import Collection

    from agent_runtime.agents import wikipedia_rag_agent
    from agent_runtime.mcp_servers.wikipedia import build_server
    from agent_runtime.workflows import mcp_rag_chat

    monkeypatch.setattr(wikipedia_rag_agent, "open_index", lambda _: Collection())

    @asynccontextmanager
    async def open_fixture(parameters):
        """Keep the same discovered MCP tool alive for either frontend."""
        async with MCPAdapter(build_server(Collection())) as adapter:
            yield create_deep_agent(**parameters, tools=await adapter.list_tools())

    monkeypatch.setattr(mcp_rag_chat, "open_agent", open_fixture)


def run_input(thread_id, prompt, attachments=None):
    """Use the standard HttpAgent body with application-owned forwarded input."""
    return {
        "threadId": thread_id,
        "runId": str(uuid4()),
        "state": {},
        "messages": [{"id": str(uuid4()), "role": "user", "content": prompt}],
        "tools": [],
        "context": [],
        "forwardedProps": {"prompt": prompt, "attachments": attachments or []},
    }


def events(response):
    """Decode SSE only in the test; the real browser uses the official SDK."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def semantics(messages):
    """Compare content and tool choices, excluding random message/block IDs."""
    result = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            content = [
                {k: v for k, v in block.items() if k != "id"}
                if isinstance(block, dict)
                else block
                for block in content
            ]
        calls = [
            (call["name"], call["args"]) for call in getattr(message, "tool_calls", [])
        ]
        result.append((message.type, content, getattr(message, "name", None), calls))
    return result


@pytest.mark.parametrize("sample", list(SAMPLES))
def test_console_and_sse_sample_parity(
    sample, tmp_path, retrieval_fixture, monkeypatch
):
    """Identical prompts and /attach context must retain identical tool history."""
    if sample.endswith("_langfuse"):
        # Real SDK ancestry has a separate in-memory exporter test. Isolate
        # client parity from remote authentication and ingestion here.
        from contextlib import nullcontext

        from langchain_core.callbacks import BaseCallbackHandler

        from agent_runtime.harness import langfuse_runtime

        class RecordingFixture:
            def __init__(self, *, settings=None):
                """Accept catalog configuration without contacting a tracing server."""
                self.settings = settings

            callback = BaseCallbackHandler()

            def scope(self, *args):
                return nullcontext()

            def close(self):
                pass

        monkeypatch.setattr(langfuse_runtime, "LangfuseCapture", RecordingFixture)
    answers = []
    if sample == "file_approval":
        answers = ["approve", "approve"]
    elif sample == "quote_request":
        from samples.quote_request.sample import ANSWERS

        answers = list(ANSWERS)
    prompts = sample_info(sample)["prompts"]
    file = tmp_path / "context.txt"
    file.write_text("Human-supplied context.\n", encoding="utf-8")
    commands = iter([f"/attach {file}", *prompts, *answers, "/quit"])
    console = ConsoleClient(read=lambda _: next(commands), write=lambda _: None)
    graph, _, _ = create_sample(sample, False)
    expected = Conversation(graph, console).invoke({}, config={})
    created = []

    def factory(selected, live):
        """Observe the session factory without altering graph execution."""
        result = create_sample(selected, live)
        created.append(result[0])
        return result

    app = create_app(directory=tmp_path / "reports", factory=factory)
    with TestClient(app, base_url="http://localhost") as client:
        thread = client.post("/api/sessions", json={"sample": sample}).json()[
            "threadId"
        ]
        web_answers = iter(answers)
        for index, prompt in enumerate(prompts):
            data = run_input(
                thread,
                prompt,
                [{"name": file.name, "content": file.read_text()}]
                if index == 0
                else [],
            )
            received = events(client.post(f"/api/chat/{thread}", json=data))
            while received[-1].get("outcome", {}).get("type") == "interrupt":
                pending = received[-1]["outcome"]["interrupts"]
                resume_data = run_input(thread, "")
                resume_data["resume"] = [
                    {
                        "interruptId": item["id"],
                        "status": "resolved",
                        "payload": next(web_answers),
                    }
                    for item in pending
                ]
                received = events(client.post(f"/api/chat/{thread}", json=resume_data))
                data = resume_data
            assert received[-1]["type"] == "RUN_FINISHED", received
            snapshot = [
                event for event in received if event["type"] == "MESSAGES_SNAPSHOT"
            ][-1]
            assert [m for m in snapshot["messages"] if not m.get("subagentRunId")][-1][
                "role"
            ] == "assistant"
            if sample == "subagent_chat":
                started = [
                    event for event in received if event["type"] == "SUBAGENT_STARTED"
                ]
                assert started
                assert started[0].get("parentToolCallId")
                assert any(
                    message.get("subagentRunId") for message in snapshot["messages"]
                )
            run = json.loads((tmp_path / "reports" / sample / "run.json").read_text())
            assert run["status"] == "ok"
            assert "Human-supplied context" not in json.dumps(run)
        # Read the server-owned native checkpoint. The browser does not own or
        # replay model/tool histories; the library reconstructs them per thread.
        session = app.state.sessions[thread]

        async def saved_messages():
            """Let LangGraph reconstruct delta checkpoints through its public API."""
            async with open_graph(session.workflow) as graph:
                graph.checkpointer = session.checkpointer
                state = await graph.aget_state({"configurable": {"thread_id": thread}})
                return state.values["messages"]

        actual = asyncio.run(saved_messages())
        assert semantics(actual) == semantics(expected["messages"])
        assert client.delete(f"/api/sessions/{thread}").status_code == 204
        assert (
            client.post(
                f"/api/chat/{thread}", json=run_input(thread, "again")
            ).status_code
            == 409
        )


class GatedModel(BaseChatModel):
    """Publish a real chat-model chunk, then wait for HTTP/graph cancellation."""

    cancelled: bool = False

    @property
    def _llm_type(self):
        """Name this synthetic model honestly in traces."""
        return "gated-test-model"

    def _generate(self, *args, **kwargs):
        """This fixture intentionally requires the native async stream path."""
        raise NotImplementedError

    async def _astream(self, *args, **kwargs):
        """Never finish naturally, so buffered responses cannot pass the test."""
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content="First chunk"))
            await asyncio.Event().wait()
        finally:
            self.cancelled = True


def gated_graph(model):
    """Build an ordinary native graph, with no knowledge of HTTP or AG-UI."""

    async def answer(state):
        """Let the model's standard callbacks expose native streaming chunks."""
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def test_native_adapter_streams_before_completion_and_cancels(tmp_path):
    """Prove streaming and teardown without a custom socket or token bridge."""
    model = GatedModel()
    session = WebConversation(
        gated_graph(model),
        "demo",
        "gated",
        sample="simple_chat",
        live=True,
        directory=tmp_path / "simple_chat",
    )

    async def exercise():
        first_chunk = asyncio.Event()
        data = RunAgentInput(
            thread_id=str(uuid4()),
            run_id=str(uuid4()),
            state={},
            messages=[UserMessage(id="u", content="hello")],
            tools=[],
            context=[],
            forwarded_props={},
        )

        async def consume():
            """Signal real delivery while keeping the run in progress."""
            async for event in session.run(data):
                if event.type == "TEXT_MESSAGE_CONTENT":
                    assert event.delta == "First chunk"
                    first_chunk.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_chunk.wait(), timeout=10)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.cancelled
        assert not session.usable
        saved = json.loads((tmp_path / "simple_chat" / "run.json").read_text())
        assert saved["status"] == "error"

    asyncio.run(exercise())


def test_http_validation_and_scripted_boundary(tmp_path):
    """Reject unsupported inputs before any model work and expose no sockets."""
    app = create_app(directory=tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/samples").json()["live"] is False
        assert (
            client.post("/api/sessions", json={"sample": "../../secret"}).status_code
            == 404
        )
        assert (
            client.post(
                "/api/sessions",
                json={"sample": "simple_chat"},
                headers={"origin": "https://foreign.example"},
            ).status_code
            == 403
        )
        thread = client.post("/api/sessions", json={"sample": "simple_chat"}).json()[
            "threadId"
        ]
        assert (
            client.post(
                f"/api/chat/{thread}", json=run_input(thread, "off-script")
            ).status_code
            == 422
        )
        bad = run_input(thread, sample_info("simple_chat")["prompts"][0])
        bad["runId"] = "../private"
        assert client.post(f"/api/chat/{thread}", json=bad).status_code == 422
        assert not any(
            route.__class__.__name__ == "APIWebSocketRoute" for route in app.routes
        )


def test_angular_launcher_resolves_package_without_building_graph(
    tmp_path, monkeypatch
):
    """The discovered catalog reaches the listener before any model construction."""
    import sys

    from agent_runtime.harness import app as launcher_app
    from agent_runtime.harness import settings as launch_settings
    from agent_runtime.web import server

    settings = SimpleNamespace(output=tmp_path, live=False)
    monkeypatch.setattr(launch_settings, "settings_for", lambda *a, **kw: settings)
    captured = {}
    monkeypatch.setattr(
        server, "start_workflow_api_listener", lambda **kw: captured.update(kw)
    )
    monkeypatch.setattr(
        catalog, "create_run", lambda *a, **kw: pytest.fail("Eager graph construction")
    )
    from agent_runtime.harness import sample_catalog

    monkeypatch.setattr(sample_catalog, "SampleCatalog", lambda: catalog)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_runtime",
            "--sample",
            "subagent_chat",
            "--client",
            "angular",
            "--port",
            "9123",
        ],
    )
    launcher_app.main()
    assert captured["catalog"] is catalog
    assert captured["sample_id"] == "subagent_chat"
    assert captured["settings"] is settings


def test_browser_approval_validates_resume_and_preserves_effects(tmp_path):
    """Wrong IDs/new turns cannot release a paused write; cancellation stops it."""
    app = create_app(directory=tmp_path / "reports")
    with TestClient(app, base_url="http://localhost") as client:
        thread = client.post("/api/sessions", json={"sample": "file_approval"}).json()[
            "threadId"
        ]
        prompt = sample_info("file_approval")["prompts"][0]
        first = events(
            client.post(f"/api/chat/{thread}", json=run_input(thread, prompt))
        )
        pending = first[-1]["outcome"]["interrupts"]
        target = app.state.sessions[thread].workflow.output_file
        assert not target.exists()
        assert (
            client.post(
                f"/api/chat/{thread}", json=run_input(thread, prompt)
            ).status_code
            == 409
        )
        data = run_input(thread, "")
        data["resume"] = [
            {"interruptId": "wrong", "status": "resolved", "payload": "approve"}
        ]
        assert client.post(f"/api/chat/{thread}", json=data).status_code == 422
        assert not target.exists()
        data["resume"][0]["interruptId"] = pending[0]["id"]
        second = events(client.post(f"/api/chat/{thread}", json=data))
        assert "Reviewed by" in target.read_text()
        assert "Next step" not in target.read_text()
        data = run_input(thread, "")
        data["resume"] = [
            {
                "interruptId": second[-1]["outcome"]["interrupts"][0]["id"],
                "status": "resolved",
                "payload": "cancel",
            }
        ]
        done = events(client.post(f"/api/chat/{thread}", json=data))
        assert done[-1]["type"] == "RUN_FINISHED"
        assert not app.state.sessions[thread].pending
        assert "Next step" not in target.read_text()
        assert client.get(f"/api/sessions/{thread}/output").text == target.read_text()
        assert client.delete(f"/api/sessions/{thread}").status_code == 204
        assert not target.exists()


@pytest.mark.parametrize("sample", ["simple_chat", "tool_chat"])
@pytest.mark.parametrize("capture_content", [False, True])
def test_retained_context_includes_latest_answer_without_persisting_content(
    tmp_path, capture_content, sample
):
    """Preview includes the final answer and tools; reports keep their opt-in."""
    from pathlib import Path

    from reporting.context import context_capacity
    from reporting.pricing import load_prices
    from reporting.schema import Run

    prices = load_prices(Path("models.json"))
    app = create_app(directory=tmp_path, prices=prices, capture_content=capture_content)
    with TestClient(app, base_url="http://localhost") as client:
        thread = client.post("/api/sessions", json={"sample": sample}).json()[
            "threadId"
        ]
        received = events(
            client.post(
                f"/api/chat/{thread}",
                json=run_input(
                    thread,
                    sample_info(sample)["prompts"][0],
                    [{"name": "unicode.txt", "content": "Café 🌍"}],
                ),
            )
        )
    metadata = next(
        event["value"] for event in received if event.get("name") == "retained_context"
    )
    run = Run.model_validate_json((tmp_path / sample / "run.json").read_text())
    latest = max(
        (step for step in run.steps if step.kind == "model"),
        key=lambda step: step.start_ns,
    )
    assert (
        metadata["utilization"]["capacity"]
        == context_capacity(latest.provider, latest.model, prices)["capacity"]
    )
    assert metadata["utilization"]["illustrative"] is True
    assert metadata["bytes"] > 0
    assert metadata["messages"][-1]["role"] == "ai"
    assert metadata["messages"][-1]["content"]
    # Native DeepAgents includes framework tools; the tool lesson adds echo_tool.
    if sample == "simple_chat":
        assert "An agent observes" in metadata["messages"][-1]["content"]
        assert "read_file" in json.dumps(metadata["tools"])
    else:
        assert "echo_tool" in json.dumps(metadata["tools"])
    assert metadata["bytes"] == len(
        json.dumps(
            {"messages": metadata["messages"], "tools": metadata["tools"]},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    if not capture_content:
        assert latest.request == []
        assert "Café" not in run.model_dump_json()


@pytest.mark.parametrize("default_live", [False, True])
def test_session_mode_selects_factory_and_preserves_launcher_default(
    tmp_path, default_live
):
    """Selection reaches the factory; omitted mode keeps launcher compatibility.

    A deterministic factory verifies dispatch without calling a paid provider.
    Strict booleans prevent strings such as "false" selecting an unintended mode.
    """
    modes = []

    def factory(sample, live):
        modes.append(live)
        return create_sample(sample, False)

    app = create_app(directory=tmp_path, factory=factory, default_live=default_live)
    with TestClient(app, base_url="http://localhost") as client:
        for selected in [None, True, False]:
            body = {"sample": "simple_chat"}
            if selected is not None:
                body["live"] = selected
            response = client.post("/api/sessions", json=body)
            assert response.status_code == 200
            session = app.state.sessions[response.json()["threadId"]]
            assert session.live is (default_live if selected is None else selected)
        assert (
            client.post(
                "/api/sessions", json={"sample": "simple_chat", "live": "false"}
            ).status_code
            == 422
        )
    assert modes == [default_live, True, False]


def test_real_mode_configuration_failure_does_not_fall_back(tmp_path):
    """A provider setup failure remains an error, never a scripted answer."""
    modes = []

    def factory(sample, live):
        modes.append(live)
        if live:
            raise ValueError("Missing provider configuration")
        return create_sample(sample, False)

    app = create_app(directory=tmp_path, factory=factory)
    with TestClient(app, base_url="http://localhost") as client:
        assert (
            client.post(
                "/api/sessions", json={"sample": "simple_chat", "live": True}
            ).status_code
            == 503
        )
        assert not app.state.sessions
        assert (
            client.post(
                "/api/sessions", json={"sample": "simple_chat", "live": False}
            ).status_code
            == 200
        )
    assert modes == [True, False]


@pytest.mark.parametrize(
    "sample", ["simple_chat", "claims_context_managed", "subagent_chat", "review_loop"]
)
def test_retained_preview_matches_next_entry_call_without_new_request(tmp_path, sample):
    """Compare preview contents with real next-turn capture, including managed purges.

    Child agent messages must not inflate the parent's context. Managed claims
    must drop invalidated history even while that history stays in the chat UI.
    """
    from reporting.schema import Run

    app = create_app(directory=tmp_path, capture_content=True)
    with TestClient(app, base_url="http://localhost") as client:
        thread = client.post("/api/sessions", json={"sample": sample}).json()[
            "threadId"
        ]
        prompts = sample_info(sample)["prompts"]
        previous = None
        for prompt in prompts:
            received = events(
                client.post(f"/api/chat/{thread}", json=run_input(thread, prompt))
            )
            assert received[-1]["type"] == "RUN_FINISHED"
            run = Run.model_validate_json((tmp_path / sample / "run.json").read_text())
            first = min(
                (step for step in run.steps if step.kind == "model"),
                key=lambda step: step.start_ns,
            )
            if previous is not None:
                # The only newly supplied content is this turn's user message.
                actual = [
                    m
                    for m in first.request
                    if not (m["role"] == "human" and m["content"] == prompt)
                ]
                assert previous["messages"] == actual
            previous = next(
                event["value"]
                for event in received
                if event.get("name") == "retained_context"
            )
            if sample == "claims_context_managed" and prompt == prompts[2]:
                serialized = json.dumps(previous["messages"])
                assert (
                    "Claim snapshots and prior discussion have been removed"
                    in serialized
                )
                calls = [
                    call["name"]
                    for message in previous["messages"]
                    for call in message["tool_calls"]
                ]
                assert "read_claim" not in calls
                assert "read_policy" in calls
