"""Verify console input sources and test clients use the conversation contract.

Inject terminal input and scripted models to exercise attachments, retained
history, and approval interrupts without provider calls. File fixtures contain
text only; these checks do not claim PDF/image support or live model quality.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import pytest
from fixtures.mock_client import MockClient
from langchain_core.messages import AIMessage

from agent_runtime.agents.chat_agent import build_agent
from agent_runtime.harness.console_client import ConsoleClient
from agent_runtime.harness.conversation import Attachment, Conversation, Request
from agent_runtime.harness.simulated_model import SimulatedModel
from samples.simple_chat.sample import CONVERSATION


@pytest.mark.parametrize("exit_kind", ["quit", "eof", "sigint"])
def test_console_exit_inside_async_runner_saves_report(tmp_path, exit_kind):
    """Real SIGINT at input must exit as cleanly as /quit and EOF.

    Raising KeyboardInterrupt directly misses asyncio.Runner's SIGINT handler:
    it cancels the task while synchronous input continues waiting for a line.
    Exercise the actual signal and recorder together, without a provider call.
    """
    import json
    import signal

    from reporting.execute_runnable import execute_runnable
    from reporting.pricing import Prices

    original_handler = signal.getsignal(signal.SIGINT)

    def read(_):
        """Reproduce a signal followed by the line that used to unblock input."""
        if exit_kind == "sigint":
            signal.raise_signal(signal.SIGINT)
        if exit_kind == "eof":
            raise EOFError
        return "/quit"

    directory = tmp_path / "report"
    result = execute_runnable(
        Conversation(None, ConsoleClient(read=read)),
        {},
        directory,
        Prices(as_of="2026-09-22", note="No model calls", models={}),
        provider="test",
        model="test",
    )
    assert result is None
    assert signal.getsignal(signal.SIGINT) == original_handler
    assert (directory / "report.html").exists()
    # No turns means no recorded spans; the exporter truthfully marks this
    # empty report incomplete, but exiting must not mark it as an error.
    assert json.loads((directory / "run.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    "payload, expected",
    [({"kind": "approval"}, "cancel"), ({"kind": "question"}, "/cancel")],
)
def test_console_sigint_at_workflow_prompt_does_not_cancel_task(payload, expected):
    """Ctrl-C answers a workflow pause using its existing cancellation contract."""
    import asyncio
    import signal

    def read(_):
        """Use a real signal so the runner's cancellation behavior is exercised."""
        signal.raise_signal(signal.SIGINT)
        return "unexpected input"

    async def answer():
        """A later await must remain usable after the prompt handles Ctrl-C."""
        handler = signal.getsignal(signal.SIGINT)
        result = ConsoleClient(read=read, write=lambda _: None).answer(payload)
        assert signal.getsignal(signal.SIGINT) == handler
        await asyncio.sleep(0)
        return result

    assert asyncio.run(answer()) == expected


# Static requests must attach file context only to the intended turn
# while retaining assistant history for the follow-up request.
def test_static_file_context_and_follow_up():
    client = MockClient(
        [
            Request("Read this note", (Attachment("note.txt", "The limit is 42."),)),
            Request("What is the limit?"),
        ]
    )
    result = Conversation(
        build_agent({"model": SimulatedModel(conversation=CONVERSATION)}), client
    ).invoke({}, {})
    assert len(client.results) == 2
    assert len(result["messages"]) == 4
    assert "note.txt\nThe limit is 42." in result["messages"][0].content
    assert result["messages"][2].content == "What is the limit?"
    assert (
        result["messages"][-1].usage_metadata["input_token_details"]["cache_read"] > 0
    )


# Console command parsing must carry attachments into the next model
# request without leaking them into unrelated commands.
def test_console_commands_keep_attachments_for_next_turn(tmp_path):
    note = tmp_path / "my note.txt"
    note.write_text("Some evidence", encoding="utf-8")
    entries = iter(
        [
            "/send",
            "",
            f"/attach {note}",
            "/attach /no/such/file",
            "/typo",
            "Explain it",
            "Follow up",
            "/quit",
        ]
    )
    output = []
    client = ConsoleClient(read=lambda _: next(entries), write=output.append)
    first = client.receive()
    assert first.prompt == "Explain it"
    assert first.files == (Attachment("my note.txt", "Some evidence"),)
    assert client.receive() == Request("Follow up")
    assert client.receive() is None
    assert any("Cannot attach" in line for line in output)


# File-only input and EOF are valid terminal paths and must terminate
# cleanly without making an unnecessary model call.
def test_console_file_only_and_eof(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("Input data")
    entries = iter([f"/attach {note}", "/send"])

    def read(_):
        # Make an exhausted scripted terminal behave like the user closing stdin.
        # Ignore the input prompt and return the next entry; convert iterator exhaustion
        # to EOFError because that is the console client termination contract.
        try:
            return next(entries)
        except StopIteration:
            raise EOFError from None

    client = ConsoleClient(read=read, write=lambda _: None)
    assert client.receive().content() == "\n\nAttached file: note.txt\nInput data"
    assert client.receive() is None


# Response order and approval interruption define the session contract;
# this guards against consuming the next request too early.
def test_respond_before_next_request_and_resume_preserves_turn():
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    def ask(state, config):
        assert config["metadata"]["report_turn"] == 1
        assert config["metadata"]["custom"] == "preserved"
        answer = interrupt({"kind": "question", "question": "Which option?"})
        return {"messages": [AIMessage(content=answer)]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    client = MockClient([Request("First")], answer=lambda _: "Chosen option")
    result = Conversation(builder.compile(), client).invoke(
        {}, {"metadata": {"custom": "preserved"}}
    )
    assert result["messages"][-1].content == "Chosen option"
    assert len(result["messages"]) == 2
    assert len(client.results) == 1


# An empty scripted client represents a no-op session and must not
# invoke the graph or fabricate a response.
def test_empty_static_client_makes_no_model_call():
    assert Conversation(None, MockClient([])).invoke({}, {}) is None


# Console and static clients must share one real graph session so
# client choice does not change workflow state semantics.
def test_console_uses_same_real_graph_session():
    entries = iter(["Explain the workflow", "And the observation?", "/quit"])
    displayed = []
    client = ConsoleClient(read=lambda _: next(entries), write=displayed.append)
    result = Conversation(
        build_agent({"model": SimulatedModel(conversation=CONVERSATION)}), client
    ).invoke({}, {})
    assert len(result["messages"]) == 4
    assert len(displayed) == 2
    assert all(line.startswith("Assistant:") for line in displayed)
    assert (
        result["messages"][-1].usage_metadata["input_token_details"]["cache_read"] > 0
    )


def test_console_script_exhaustion_never_reads_terminal():
    """Authored prompts use normal presentation and finish without an input fallback."""
    from agent_runtime.harness.script_prompter import ScriptPrompter

    def unexpected_read(_):
        raise AssertionError("Script exhausted: terminal must not be read")

    displayed = []
    client = ConsoleClient(
        read=unexpected_read,
        write=displayed.append,
        prompter=ScriptPrompter([Request("first"), Request("second")]),
    )
    result = Conversation(
        build_agent({"model": SimulatedModel(conversation=CONVERSATION)}), client
    ).invoke({}, {})
    assert len(result["messages"]) == 4
    assert len(displayed) == 2
    assert client.receive() is None
    assert not hasattr(client, "results")


def test_console_script_can_ask_human_without_advancing_prompt():
    """Scripted requests and human interruption answers are independent inputs."""
    from agent_runtime.harness.script_prompter import ScriptPrompter

    client = ConsoleClient(
        read=lambda _: "approve",
        write=lambda _: None,
        prompter=ScriptPrompter([Request("edit"), Request("review")]),
    )
    assert client.receive().prompt == "edit"
    assert client.answer({"kind": "approval"}) == "approve"
    assert client.receive().prompt == "review"


def test_static_launch_rejects_unexpected_interrupt_and_respects_empty_prompts():
    """Unattended runs neither ask stdin nor invent approval for an unscripted pause."""
    import pytest

    from agent_runtime.harness.configure_sample_script import (
        configure_sample_script,
    )
    from agent_runtime.harness.sample_catalog import SampleCatalog

    client = ConsoleClient()
    configure_sample_script(
        client, SampleCatalog(), "simple_chat", prompts=[], script_answers=True
    )
    assert isinstance(client, ConsoleClient)
    assert client.receive() is None
    with pytest.raises(ValueError, match="No scripted response"):
        client.answer({"kind": "approval"})


def test_configure_sample_script_preserves_existing_client_and_answer_behavior():
    """Adding prompt playback must not replace the client or its human-answer policy."""
    from agent_runtime.harness.configure_sample_script import (
        configure_sample_script,
    )
    from agent_runtime.harness.sample_catalog import SampleCatalog

    client = ConsoleClient()
    assert client.prompter is None
    assert client.answer_callback is None
    client.answer_callback = lambda payload: "human choice"
    result = configure_sample_script(
        client, SampleCatalog(), "simple_chat", prompts=["one", "two"]
    )
    assert result is None
    assert client.receive().prompt == "one"
    assert client.answer({"kind": "question"}) == "human choice"
    assert client.receive().prompt == "two"
    assert client.receive() is None
