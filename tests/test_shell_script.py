"""Verify native shell execution, real artifacts, and observable command failures.

The model is scripted but the shell, filesystem, native execute tool, and async
backend path are real. Temporary directories keep test artifacts separate.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
from functools import partial

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.messages import ToolMessage

from agent_runtime.backends.shell_backend import ShellBackend
from agent_runtime.harness.sample_catalog import SampleCatalog

create_run = partial(SampleCatalog().create_run, "shell_script")
from samples.shell_script.scripted_run import USER_PROMPTS


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_execute_runs_script_and_reports_observation(asynchronous):
    """Both clients must execute the script, not merely emit a scripted success."""
    graph, _, _ = create_run(False)
    try:
        payload = {"messages": [("user", USER_PROMPTS[0])]}
        result = (
            asyncio.run(graph.ainvoke(payload))
            if asynchronous
            else graph.invoke(payload)
        )
        observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert observation.name == "execute"
        assert "Orders: 3\nTotal units: 9" in observation.content
        assert "exit code 0" in observation.content
        assert observation.content in result["messages"][-1].content
        assert graph.output_file.read_text() == "Orders: 3\nTotal units: 9\n"
    finally:
        graph.workspace.cleanup()


def test_shell_failure_preserves_stderr_and_exit_code(tmp_path):
    """A failed process is evidence the agent must see, not a Python success stub."""
    backend = ShellBackend(tmp_path)
    assert isinstance(backend, SandboxBackendProtocol)
    assert backend.id == backend.id
    assert backend.id != ShellBackend(tmp_path).id
    result = backend.execute("printf 'failure detail\\n' >&2; exit 7")
    assert result.output == "failure detail\n"
    assert result.exit_code == 7
    assert not result.truncated


def test_shell_deadline_returns_partial_output(tmp_path):
    """The local runner enforces its deadline and reports timeout as failure."""
    result = ShellBackend(tmp_path).execute("printf 'started\\n'; sleep 30", timeout=1)
    assert result.exit_code == 124
    assert "started" in result.output
    assert "timed out" in result.output


def test_shell_capture_limit_is_explicit(tmp_path):
    """Large output cannot masquerade as a complete tool observation."""
    result = ShellBackend(tmp_path).execute(
        "awk 'BEGIN { for (i=0; i<21000; i++) printf \"x\" }'"
    )
    assert result.exit_code == 0
    assert len(result.output) == 20_000
    assert result.truncated


def test_shell_file_tools_and_commands_share_the_workspace(tmp_path):
    """DeepAgent's virtual file paths and relative shell paths refer to the same files."""
    backend = ShellBackend(tmp_path)
    assert backend.write("/input.txt", "shared content").error is None
    assert backend.execute("cat input.txt", timeout=0).output == "shared content"
    with pytest.raises(ValueError, match="non-negative"):
        backend.execute("exit 0", timeout=-1)


def test_fixture_reports_real_script_failure():
    """Removing required input must reach the final answer as a failed command."""
    graph, _, _ = create_run(False)
    try:
        (graph.output_file.parent / "orders.csv").unlink()
        result = graph.invoke({"messages": [("user", USER_PROMPTS[0])]})
        assert "Command failed" in result["messages"][-1].content
        assert "Orders: 3" not in result["messages"][-1].content
    finally:
        graph.workspace.cleanup()


@pytest.mark.parametrize("live", [False, True])
def test_prepared_script_runs_directly_through_native_execute(monkeypatch, live):
    """A provider may choose the shebang form instead of explicitly invoking sh."""
    from langchain_core.messages import AIMessage

    from agent_runtime.harness import model_factory
    from agent_runtime.harness.simulated_model import ScriptedChatModel
    from samples.shell_script import scripted_run

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "./summarize.sh"},
                        "id": "direct-script",
                    }
                ],
            ),
            AIMessage(content="Finished"),
        ]
    )
    # Exercise both construction branches without making a paid provider call.
    # Only the model is substituted; workspace preparation and execution are real.
    monkeypatch.setattr(scripted_run, "make_simulated_model", lambda: model)
    monkeypatch.setattr(
        model_factory, "configured_model", lambda name: (model, "demo", "scripted-chat")
    )
    graph, _, _ = create_run(live)
    try:
        result = graph.invoke({"messages": [("user", "Run ./summarize.sh")]})
        observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert "exit code 0" in observation.content, observation.content
        assert graph.output_file.read_text() == "Orders: 3\nTotal units: 9\n"
    finally:
        graph.workspace.cleanup()
