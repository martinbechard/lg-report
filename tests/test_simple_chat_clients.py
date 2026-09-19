"""Verify that console and static clients use the same conversation contract.

Inject terminal input and scripted models to exercise attachments, retained
history, and approval interrupts without provider calls. File fixtures contain
text only; these checks do not claim PDF/image support or live model quality.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage

from lg_report.agents.chat_agent import build_agent
from lg_report.platform.console_client import ConsoleClient
from lg_report.platform.conversation import Attachment, Conversation, Request
from lg_report.platform.static_client import StaticClient
from samples.simple_chat.test_case import make_simulated_model


# Static requests must attach file context only to the intended turn
# while retaining assistant history for the follow-up request.
def test_static_file_context_and_follow_up():
    client = StaticClient(
        [
            Request("Read this note", (Attachment("note.txt", "The limit is 42."),)),
            Request("What is the limit?"),
        ]
    )
    result = Conversation(build_agent(make_simulated_model()), client).invoke({}, {})
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
def test_respond_before_next_request_and_interrupt_stops_session():
    class Graph:
        def invoke(self, inputs, config):
            # Return a synthetic pause to test the conversation driver stopping rule.
            # Validate config forwarding, then supply graph-shaped messages/interrupt data.
            # This stub does not test actual LangGraph checkpoint or replay behavior.
            assert config["metadata"]["report_turn"] == 1
            assert config["metadata"]["custom"] == "preserved"
            assert config["callbacks"] == ["capture"]
            return {
                "messages": [AIMessage(content="Paused")],
                "__interrupt__": ["approval"],
            }

    client = StaticClient([Request("First"), Request("Never sent")])
    result = Conversation(Graph(), client).invoke(
        {}, {"metadata": {"custom": "preserved"}, "callbacks": ["capture"]}
    )
    assert result["__interrupt__"]
    assert len(client.results) == 1
    assert client.receive().prompt == "Never sent"


# An empty scripted client represents a no-op session and must not
# invoke the graph or fabricate a response.
def test_empty_static_client_makes_no_model_call():
    assert Conversation(None, StaticClient([])).invoke({}, {}) is None


# Console and static clients must share one real graph session so
# client choice does not change workflow state semantics.
def test_console_uses_same_real_graph_session():
    entries = iter(["Explain the workflow", "And the observation?", "/quit"])
    displayed = []
    client = ConsoleClient(read=lambda _: next(entries), write=displayed.append)
    result = Conversation(build_agent(make_simulated_model()), client).invoke({}, {})
    assert len(result["messages"]) == 4
    assert len(displayed) == 2
    assert all(line.startswith("Assistant:") for line in displayed)
    assert (
        result["messages"][-1].usage_metadata["input_token_details"]["cache_read"] > 0
    )
