"""Verify workflow dependency ownership without paid model requests.

These checks protect policy retention, explicit graph-boundary persistence, and
workflow-owned backend construction after consolidating agent factories.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime.backends.file_access_backend import FileAccessBackend
from agent_runtime.backends.shell_backend import ShellBackend
from agent_runtime.harness.simulated_model import ScriptedChatModel
from agent_runtime.workflows import file_approval, shell_script


def test_file_workflow_supplies_backend_policy_and_saver(monkeypatch, tmp_path):
    """The editor gets configured dependencies, not paths from which to select access."""
    monkeypatch.setattr(
        file_approval,
        "build_agent",
        lambda parameters: SimpleNamespace(parameters=parameters),
    )
    saver = InMemorySaver()
    graph = file_approval.build_workflow(
        ScriptedChatModel(responses=[]),
        source=tmp_path / "source",
        target=tmp_path / "target",
        checkpointer=saver,
    )
    parameters = graph.parameters
    assert parameters["checkpointer"] is saver
    assert parameters["middleware"][1].restricted_tools == {"write_file", "edit_file"}
    files = parameters["middleware"][0]
    assert isinstance(files.backend, FileAccessBackend)
    assert files.backend.paths["/target.txt"] == tmp_path / "target"
    assert {tool.name for tool in files.tools} == {
        "read_file",
        "write_file",
        "edit_file",
    }


def test_shell_workflow_selects_execution_backend(monkeypatch, tmp_path):
    """Native execute support comes from workflow configuration, not role construction."""
    monkeypatch.setattr(
        shell_script,
        "build_agent",
        lambda parameters: SimpleNamespace(parameters=parameters),
    )
    graph = shell_script.build_workflow(
        ScriptedChatModel(responses=[]), working_directory=tmp_path
    )
    parameters = graph.parameters
    assert isinstance(parameters["backend"], ShellBackend)
    assert parameters.get("middleware", ()) == ()


def test_native_delegation_retains_parent_approval():
    """Passing native options through the dictionary must protect child tool calls."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from agent_runtime.agents.delegating_parent import build_agent

    writes = []

    @tool
    def record_note(text: str) -> str:
        """Record a note only after the workflow's approval policy permits it."""
        writes.append(text)
        return text

    parent = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": "delegate",
                        "args": {
                            "subagent_type": "isolated-subagent",
                            "description": "Record a note",
                        },
                    }
                ],
            )
        ]
    )
    child = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record_note",
                        "id": "write",
                        "args": {"text": "Pending approval"},
                    }
                ],
            )
        ]
    )
    graph = build_agent(
        {
            "model": parent,
            "interrupt_on": {"record_note": True},
            "checkpointer": InMemorySaver(),
        },
        {
            "name": "isolated-subagent",
            "description": "Records notes",
            "system_prompt": "Record the requested note",
            "model": child,
            "tools": [record_note],
        },
    )
    result = graph.invoke(
        {"messages": [("user", "Record a note")]},
        {"configurable": {"thread_id": "native-child-approval"}},
    )
    assert result["__interrupt__"]
    assert writes == []
