"""Protect discovery and per-conversation input ownership without provider calls.

New sample metadata must reach both clients without a central Python registry.
Console commands must never become model prompts or interruption answers.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

import pytest

from agent_runtime.harness.console_application import ConsoleApplication
from agent_runtime.harness.console_client import ConsoleClient
from agent_runtime.harness.conversation import Request
from agent_runtime.harness.sample_catalog import SampleCatalog
from agent_runtime.harness.script_prompter import ScriptPrompter


def register(root, folder, sample_id):
    """Create metadata independently of the catalog's implementation registry."""
    directory = root / folder
    directory.mkdir()
    (directory / "sample.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": sample_id,
                        "name": "Discovered lesson",
                        "description": "A new lesson",
                        "workflow": "agent_runtime.workflows.simple_chat:build_workflow",
                        "scripted_run": "samples.simple_chat.scripted_run",
                    }
                ]
            }
        )
    )
    return directory


def test_new_folder_is_discovered_without_imports_and_reaches_http(
    tmp_path, monkeypatch
):
    """Listing reads metadata only; HTTP then loads prompts for the discovered ID."""
    from fastapi.testclient import TestClient

    from agent_runtime.harness import sample_catalog
    from agent_runtime.web.server import create_app

    register(tmp_path, "new_lesson", "custom_lesson")
    with monkeypatch.context() as patch:
        patch.setattr(sample_catalog, "import_module", lambda name: pytest.fail(name))
        catalog = SampleCatalog(tmp_path)
        assert catalog.get("custom_lesson").name == "Discovered lesson"
    with TestClient(
        create_app(catalog=catalog, directory=tmp_path / "reports"),
        base_url="http://localhost",
    ) as client:
        response = client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()
        assert data["defaultSample"] == "custom_lesson"
        assert data["samples"][0]["id"] == "custom_lesson"
        assert data["samples"][0]["prompts"]
        created = client.post("/api/sessions", json={"sample": "custom_lesson"})
        assert created.status_code == 200
        thread = created.json()["threadId"]
        assert client.delete(f"/api/sessions/{thread}").status_code == 204


def test_duplicate_metadata_names_its_source(tmp_path):
    """Two folders cannot silently compete for one public sample ID."""
    register(tmp_path, "a", "lesson")
    register(tmp_path, "b", "lesson")
    with pytest.raises(ValueError, match=r"b/sample.json.*Duplicate"):
        SampleCatalog(tmp_path)


def test_sample_configuration_does_not_leak(tmp_path, monkeypatch):
    """Switching samples must not inherit the first selection's dotenv model."""
    first = register(tmp_path, "a", "first")
    second = register(tmp_path, "b", "second")
    (first / ".env").write_text("LG_MODEL=first-model\n")
    (second / ".env").write_text("LG_MODEL=second-model\n")
    monkeypatch.delenv("LG_MODEL", raising=False)
    from agent_runtime.harness.model_config import configured_identity

    catalog = SampleCatalog(tmp_path)
    assert (
        configured_identity(settings=catalog.configuration("first"))[1] == "first-model"
    )
    assert (
        configured_identity(settings=catalog.configuration("second"))[1]
        == "second-model"
    )
    monkeypatch.setenv("LG_MODEL", "explicit-shell-model")
    assert (
        configured_identity(settings=catalog.configuration("first"))[1]
        == "explicit-shell-model"
    )


def test_script_prompter_exhausts_and_resets_independently():
    """Peeking does not consume, and exhausted scripts cannot accidentally recycle."""
    first = ScriptPrompter([Request("one"), Request("two")])
    other = ScriptPrompter([Request("one")])
    assert first.peek().prompt == first.next().prompt == "one"
    assert first.next().prompt == "two"
    assert first.next() is None
    assert first.next() is None
    assert other.next().prompt == "one"
    first.reset()
    assert first.next().prompt == "one"


def test_console_commands_switch_fresh_clients_and_do_not_reach_workflow(monkeypatch):
    """Listing/invalid selection stay local; /new and /sample close each session."""
    catalog = SampleCatalog()
    lines = iter(
        [
            "/samples",
            "/sample missing",
            "first",
            "/sample tool_chat",
            "second",
            "/new",
            "/quit",
        ]
    )
    output, sessions = [], []

    def run_session(selected, client):
        prompts = []
        while request := client.receive():
            prompts.append(request.prompt)
        sessions.append((selected, client, prompts))

    from types import SimpleNamespace

    application = ConsoleApplication(
        catalog,
        SimpleNamespace(client="console"),
        read=lambda _: next(lines),
        write=output.append,
    )
    monkeypatch.setattr(application, "run_session", run_session)
    application.run("simple_chat")
    assert [(id, prompts) for id, _, prompts in sessions] == [
        ("simple_chat", ["first"]),
        ("tool_chat", ["second"]),
        ("tool_chat", []),
    ]
    assert len({id(client) for _, client, _ in sessions}) == 3
    assert any("Unknown sample" in text for text in output)
    assert any("subagent_chat" in text for text in output)


def test_interruption_answer_bypasses_commands_and_structured_run_returns_to_menu():
    """Slash text during an answer belongs to the workflow, not sample selection."""
    catalog = SampleCatalog()
    lines = iter(["/sample tool_chat", "/samples", "/sample simple_chat"])
    client = ConsoleClient(
        catalog=catalog,
        sample_id="quote_request",
        read=lambda _: next(lines),
        write=lambda _: None,
    )
    from agent_runtime.harness.configure_sample_script import configure_sample_script

    configure_sample_script(client, catalog, "quote_request")
    assert client.receive() is not None
    assert client.answer({"kind": "clarification"}) == "/sample tool_chat"
    assert client.next_sample is None
    assert client.receive() is None
    assert client.next_sample == "simple_chat"


def test_file_approval_preserves_human_client_default(monkeypatch):
    """An omitted --client must keep human approval after moving CLI into the harness."""
    import sys
    from types import SimpleNamespace

    from agent_runtime.harness import app, console_application

    opened = []
    monkeypatch.setattr(sys, "argv", ["agent_runtime", "--sample", "file_approval"])
    monkeypatch.setattr(
        console_application,
        "ConsoleApplication",
        lambda *a, **kw: SimpleNamespace(run=opened.append),
    )
    app.main()
    assert opened == ["file_approval"]
    assert SampleCatalog().get("simple_chat").default_client == "static"


def test_console_session_method_clears_initial_overrides_on_sample_switch(monkeypatch):
    """Console ownership includes per-selection setup, without capturing a closure."""
    from types import SimpleNamespace

    from agent_runtime.harness import execute_conversation, settings

    initial = SimpleNamespace(
        sample="simple_chat",
        client="console",
        options={"initial": True},
        source="input",
        target="output",
        mode="initial",
        values="values.json",
        request="initial prompt",
        out="initial-report",
        env_file=None,
        show_context=False,
        public_trace=False,
    )
    prepared, runs = [], []

    def prepare(catalog, selected, args):
        prepared.append((selected, args))
        return object(), {}, None

    monkeypatch.setattr(settings, "prepare_sample", prepare)
    monkeypatch.setattr(
        execute_conversation,
        "execute_conversation",
        lambda **kwargs: runs.append(kwargs),
    )
    application = ConsoleApplication(SampleCatalog(), initial)
    client = object()
    application.run_session("tool_chat", client)
    selected, args = prepared[0]
    assert selected == "tool_chat"
    assert args.options == {}
    assert all(
        getattr(args, key) is None
        for key in ("source", "target", "mode", "values", "request", "out")
    )
    assert runs[0]["client"] is client
    assert initial.options == {"initial": True}
    assert initial.request == "initial prompt"


@pytest.mark.parametrize(
    "client_mode,live,expected_scripts",
    [
        ("static", False, 1),
        ("static", True, 1),
        ("console", True, 0),
        ("angular", False, 0),
    ],
)
def test_launcher_delegates_terminal_script_policy_to_console_application(
    monkeypatch, client_mode, live, expected_scripts
):
    """All terminal paths share preparation; only static mode scripts normal chat."""
    import sys

    from agent_runtime.harness import (
        app,
        configure_sample_script,
        execute_conversation,
        settings,
    )
    from agent_runtime.web import server

    configured, executed = [], []
    monkeypatch.setattr(settings, "prepare_sample", lambda *a: (object(), {}, None))
    monkeypatch.setattr(
        configure_sample_script,
        "configure_sample_script",
        lambda *a, **kw: configured.append(kw),
    )
    monkeypatch.setattr(
        execute_conversation, "execute_conversation", lambda **kw: executed.append(kw)
    )
    monkeypatch.setattr(server, "start_workflow_api_listener", lambda **kw: None)
    argv = ["agent_runtime", "--sample", "simple_chat", "--client", client_mode]
    if live:
        argv.append("--live")
    monkeypatch.setattr(sys, "argv", argv)
    app.main()
    assert len(configured) == expected_scripts
    assert len(executed) == (0 if client_mode == "angular" else 1)
    if configured:
        assert configured[0]["script_answers"] is True


def test_static_console_application_exits_when_script_is_exhausted(monkeypatch):
    """A static structured sample must not fall through into the catalog menu."""
    from types import SimpleNamespace

    from agent_runtime.harness import execute_conversation, settings

    args = SimpleNamespace(
        sample="file_approval",
        client="static",
        scenario="complete",
        decision="approve",
        env_file=None,
        show_context=False,
        public_trace=False,
    )
    monkeypatch.setattr(settings, "prepare_sample", lambda *a: (object(), {}, None))
    received = []

    def execute(**kwargs):
        client = kwargs["client"]
        received.append(client.receive())
        assert client.answer({"kind": "approval"}) == "approve"
        assert client.receive() is None
        assert client.receive() is None

    def unexpected_read(_):
        pytest.fail("Static run must never read the terminal")

    monkeypatch.setattr(execute_conversation, "execute_conversation", execute)
    ConsoleApplication(
        SampleCatalog(), args, read=unexpected_read, write=lambda _: None
    ).run(args.sample)
    assert len(received) == 1
    assert received[0] is not None
