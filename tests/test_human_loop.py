"""Verify real interrupt/resume boundaries and both standalone human-loop clients.

Temporary files prove that approval gates effects, not merely displayed messages.
CLI tests exercise the same reporting path as the documented sample commands.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_runtime.workflows.file_approval import build_workflow as file_workflow
from agent_runtime.workflows.quote_request import build_workflow as quote_workflow

# Each test creates its own graph/saver; this id links pause and resume calls
# within that graph, not persistence across separate tests or processes.
CONFIG = {"configurable": {"thread_id": "test"}}


# Build isolated source/target fixtures with two sequential changes
# so approval, cancellation, and concurrent-edit checks observe real files.
def files(tmp_path, mode="always-ask"):
    source = tmp_path / "source.txt"
    source.write_text("Original\n")
    target = tmp_path / "target.txt"
    return (
        source,
        target,
        {
            "source": str(source),
            "target": str(target),
            "mode": mode,
            "additions": ["First\n", "Second\n"],
        },
    )


def editor(inputs):
    """Wire the same agent and harness as the CLI with two scripted model edits."""
    from samples.file_approval.sample import make_simulated_model

    # Only the model's decisions are simulated. Reads, approval interrupts,
    # checkpoint resumes, and writes all run through the production graph.
    return file_workflow(
        make_simulated_model(inputs["additions"]),
        source=inputs["source"],
        target=inputs["target"],
        mode=inputs["mode"],
        checkpointer=InMemorySaver(),
    )


def request():
    """Supply a human request rather than workflow-authored file contents."""
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content="Append First and Second separately.")]}


def test_each_write_requires_approval(tmp_path):
    """Reads execute freely; actual model-selected writes pause before execution."""
    from langchain_core.callbacks import BaseCallbackHandler

    class Capture(BaseCallbackHandler):
        """Count model requests and executed tools across checkpoint resumes."""

        def __init__(self):
            self.models = 0
            self.tools = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.models += 1

        def on_tool_start(self, serialized, input_str, **kwargs):
            self.tools.append(serialized["name"])

    source, target, inputs = files(tmp_path)
    capture = Capture()
    config = {**CONFIG, "callbacks": [capture]}
    graph = editor(inputs)
    first = graph.invoke(request(), config)
    assert not target.exists()
    assert first["__interrupt__"][0].value["tool"] == "write_file"
    assert "First" in first["__interrupt__"][0].value["arguments"]["content"]
    assert capture.tools == ["read_file", "read_file"]
    assert capture.models == 3
    second = graph.invoke(Command(resume="approve"), config)
    assert target.read_text() == "Original\nFirst\n"
    assert second["__interrupt__"]
    assert capture.models == 5  # Only target reread and second proposal, no replay.
    invalid = graph.invoke(Command(resume="yes"), config)
    assert invalid["__interrupt__"]
    assert capture.models == 5
    assert capture.tools.count("write_file") == 1
    result = graph.invoke(Command(resume="approve"), config)
    assert result["changes"] == ["approve", "approve"]
    assert result["status"] == "completed"
    assert capture.models == 6
    assert capture.tools.count("write_file") == 1
    assert capture.tools.count("edit_file") == 1
    assert target.read_text() == "Original\nFirst\nSecond\n"
    assert source.read_text() == "Original\n"


def test_autoapprove_never_pauses(tmp_path):
    """The same model/tool loop runs without interrupts under explicit policy."""
    _, target, inputs = files(tmp_path, "autoapprove")
    result = editor(inputs).invoke(request(), CONFIG)
    assert not result.get("__interrupt__")
    assert target.read_text() == "Original\nFirst\nSecond\n"


def test_reject_then_approve_does_not_reintroduce_rejected_change(tmp_path):
    """Rejection reaches the agent as a result; its fresh read sees no write."""
    _, target, inputs = files(tmp_path)
    graph = editor(inputs)
    graph.invoke(request(), CONFIG)
    second = graph.invoke(Command(resume="reject"), CONFIG)
    assert not target.exists()
    assert "First" not in second["__interrupt__"][0].value["arguments"]["content"]
    result = graph.invoke(Command(resume="approve"), CONFIG)
    assert result["changes"] == ["reject", "approve"]
    assert target.read_text() == "Original\nSecond\n"


@pytest.mark.parametrize("first", [True, False])
def test_file_cancel_stops_future_changes(tmp_path, first):
    """Cancellation terminates the checkpoint, preserving earlier actual writes."""
    _, target, inputs = files(tmp_path)
    graph = editor(inputs)
    graph.invoke(request(), CONFIG)
    if not first:
        graph.invoke(Command(resume="approve"), CONFIG)
    result = graph.invoke(Command(resume="cancel"), CONFIG)
    assert result["status"] == "cancelled"
    assert not result.get("__interrupt__")
    assert graph.get_state(CONFIG).next == ()
    assert (not target.exists()) if first else target.read_text() == "Original\nFirst\n"


def test_changed_target_is_not_overwritten(tmp_path):
    """A paused proposal cannot overwrite an intervening human change."""
    _, target, inputs = files(tmp_path)
    graph = editor(inputs)
    graph.invoke(request(), CONFIG)
    target.write_text("Human edit\n")
    with pytest.raises(RuntimeError, match="Target changed"):
        graph.invoke(Command(resume="approve"), CONFIG)
    assert target.read_text() == "Human edit\n"


def test_in_place_edit_and_existing_output_are_reviewed(tmp_path):
    """The same gate protects edits where source and target are the same path."""
    source, _, inputs = files(tmp_path)
    inputs["target"] = str(source)
    graph = editor(inputs)
    first = graph.invoke(request(), CONFIG)
    assert first["__interrupt__"][0].value["expected_content"] == "Original\n"
    graph.invoke(Command(resume="approve"), CONFIG)
    assert source.read_text() == "Original\nFirst\n"
    graph.invoke(Command(resume="reject"), CONFIG)
    assert source.read_text() == "Original\nFirst\n"


def test_agent_can_finish_without_requesting_a_write(tmp_path):
    """The harness must not manufacture edits or approvals from the request."""
    from langchain_core.messages import AIMessage

    from agent_runtime.harness.simulated_model import ScriptedChatModel

    source, target, _ = files(tmp_path)
    graph = file_workflow(
        ScriptedChatModel(responses=[AIMessage(content="No edit needed.")]),
        source=source,
        target=target,
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(request(), CONFIG)
    assert result["status"] == "completed"
    assert not result.get("__interrupt__")
    assert not target.exists()


# Field presence alone must not bypass a scripted clarification decision.
# Each human answer must reach the next assessment in the retained context.
def test_quote_model_questions_resolve_multiple_issues():
    """Complete fields still pause; each human answer reaches the next model call."""
    from langchain_core.callbacks import BaseCallbackHandler

    from agent_runtime.agents.quote_interpreter import SYSTEM_PROMPT
    from samples.quote_request.sample import (
        ANSWERS,
        DECISIONS,
        INITIAL_VALUES,
        make_simulated_model,
    )

    class Capture(BaseCallbackHandler):
        """Observe actual model inputs to prove clarification context is retained."""

        def __init__(self):
            # Keep observations isolated to this clarification run.
            # Separate lists distinguish user context, system instructions, and agent spans.
            self.prompts = []
            self.system_prompts = []
            self.agent_runs = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            # Observe whether the next assessment receives all prior human answers.
            # LangChain supplies a batch of message lists; this fixture makes one request,
            # whose first message is the system instruction and last is the user payload.
            self.prompts.append(messages[0][-1].content)
            self.system_prompts.append(messages[0][0].content)

        def on_chain_start(self, serialized, inputs, *, name=None, **kwargs):
            """Record role spans so the test proves workflow-to-agent delegation."""
            if name == "quote_interpreter":
                self.agent_runs.append(name)

    capture = Capture()
    config = {**CONFIG, "callbacks": [capture]}
    graph = quote_workflow(make_simulated_model(), checkpointer=InMemorySaver())
    result = graph.invoke({"values": INITIAL_VALUES}, config)
    # These decisions are scripted, not live semantic judgments. Resume replays
    # ask from its start; only after its answer update does routing run assess
    # again. Counting model starts below catches accidental reassessment replay.
    for decision, answer in zip(DECISIONS, ANSWERS):
        assert result["__interrupt__"][0].value["question"] == decision["text"]
        assert result["request"] is None
        result = graph.invoke(Command(resume=answer), config)
    assert result["status"] == "completed"
    assert len(result["conversation"]) == 3
    assert all(answer in capture.prompts[-1] for answer in ANSWERS)
    assert len(capture.prompts) == 4  # Resume must not repeat the previous model call.
    # The extracted agent supplies its instructions and remains visible to the
    # caller's recorder on every assessment, including after human interrupts.
    assert capture.system_prompts == [SYSTEM_PROMPT] * 4
    # Each assessment records the domain adapter and its native agent graph.
    # Both carry the role name; the inner graph owns the actual model call.
    assert capture.agent_runs == ["quote_interpreter"] * 8
    assert not result.get("__interrupt__")


# A clear request may complete immediately; the workflow must follow
# the model decision rather than enforce a fixed question count.
def test_quote_model_can_complete_without_questions():
    """Routing follows the model, not mandatory fields or a fixed question count."""
    from samples.quote_request.sample import make_simulated_model

    decision = {
        "action": "complete",
        "reason": "Scope is clear.",
        "text": "Print 20 identical A4 posters; customer supplies artwork.",
    }
    result = quote_workflow(
        make_simulated_model([decision]), checkpointer=InMemorySaver()
    ).invoke({"values": {"description": decision["text"]}}, CONFIG)
    assert result["status"] == "completed"
    assert not result.get("__interrupt__")


# A vague human answer can leave the request unresolved, so the model
# gets another turn with the first answer retained in conversation state.
def test_quote_unclear_answer_can_trigger_another_question():
    """The human's first reply does not automatically resolve the model's concern."""
    from samples.quote_request.sample import (
        DECISIONS,
        INITIAL_VALUES,
        make_simulated_model,
    )

    graph = quote_workflow(
        make_simulated_model([DECISIONS[0], DECISIONS[0]]), checkpointer=InMemorySaver()
    )
    graph.invoke({"values": INITIAL_VALUES}, CONFIG)
    result = graph.invoke(Command(resume="Whatever is best"), CONFIG)
    assert result["__interrupt__"]
    assert result["request"] is None
    assert result["conversation"][0]["answer"] == "Whatever is best"


# Explicit cancellation must clear partial values, conversation, and request
# from the returned state. This does not test deletion of checkpoint history.
def test_quote_abandon_discards_partial_request():
    from samples.quote_request.sample import INITIAL_VALUES, make_simulated_model

    graph = quote_workflow(make_simulated_model(), checkpointer=InMemorySaver())
    graph.invoke({"values": INITIAL_VALUES}, CONFIG)
    result = graph.invoke(Command(resume="/cancel"), CONFIG)
    assert result["status"] == "cancelled"
    assert result["request"] is None
    assert result["values"] == {}
    assert result["conversation"] == []


# Malformed model output is a contract failure and must not be treated
# as a completed quote merely because the model returned text.
def test_quote_malformed_model_output_cannot_complete():
    """An adapter failure must not silently turn uncertainty into a finished quote."""
    from langchain_core.messages import AIMessage

    from agent_runtime.harness.simulated_model import ScriptedChatModel

    model = ScriptedChatModel(responses=[AIMessage(content="Looks fine")])
    from langgraph.errors import GraphRecursionError

    with pytest.raises(GraphRecursionError):
        quote_workflow(model, checkpointer=InMemorySaver()).invoke(
            {"values": {}}, {**CONFIG, "recursion_limit": 6}
        )


@pytest.mark.parametrize(
    "sample,extra,stdin",
    [
        ("file_approval", ["--mode", "autoapprove"], ""),
        ("file_approval", [], "approve\napprove\n"),
        ("file_approval", [], "reject\ncancel\n"),
        ("quote_request", ["--client", "static"], ""),
        ("quote_request", ["--client", "static", "--scenario", "cancel"], ""),
    ],
)
# Execute documented CLI entry points from an external working
# directory to verify packaging, reporting, pricing, and approval behavior.
def test_standalone_samples(tmp_path, sample, extra, stdin):
    root = Path(__file__).resolve().parents[1]
    rate = tmp_path / "fx.json"
    rate.write_text(json.dumps({"rate": "0.871", "date": "2026-09-18"}))
    output = tmp_path / "report"
    args = [
        sys.executable,
        "-m",
        "agent_runtime", "--demo", "--sample", sample,
        "--prices",
        str(root / "models.json"),
        "--fx-file",
        str(rate),
        "--out",
        str(output),
        *extra,
    ]
    if sample == "file_approval":
        source, target, _ = files(tmp_path)
        args += ["--source", str(source), "--target", str(target), "--client", "console"]
    result = subprocess.run(
        args,
        cwd=tmp_path,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "report.html").exists()
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "ok"
    assert (
        '"status": "cancelled"' in result.stdout
        if "cancel" in stdin or "cancel" in extra
        else '"status": "completed"' in result.stdout
    )
    if sample == "file_approval":
        assert (
            (not target.exists())
            if "reject" in stdin
            else "Next step" in target.read_text()
        )


@pytest.mark.parametrize("cancel_batch", [False, True])
def test_batched_restricted_calls_are_gated(tmp_path, cancel_batch):
    """Every model-selected write is reviewed before the tools node executes."""
    from langchain_core.messages import AIMessage

    from agent_runtime.harness.simulated_model import ScriptedChatModel

    source, target, _ = files(tmp_path)
    # Deliberately bypass the prompt's request for sequential calls to prove
    # authorization is enforced by the harness even for a model-produced batch.
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/target.txt", "content": "First"},
                        "id": "first",
                    },
                    {
                        "name": "write_file",
                        "args": {"file_path": "/target.txt", "content": "Second"},
                        "id": "second",
                    },
                ],
            ),
            AIMessage(content="Finished."),
        ]
    )
    graph = file_workflow(
        model, source=source, target=target, checkpointer=InMemorySaver()
    )
    first = graph.invoke(request(), CONFIG)
    assert first["__interrupt__"]
    second = graph.invoke(
        Command(resume="approve" if cancel_batch else "reject"), CONFIG
    )
    assert second["__interrupt__"]
    assert not target.exists()  # The first reviewed call has not run yet.
    result = graph.invoke(
        Command(resume="cancel" if cancel_batch else "approve"), CONFIG
    )
    assert graph.get_state(CONFIG).next == ()
    if cancel_batch:
        assert result["status"] == "cancelled"
        assert not target.exists()
    else:
        assert target.read_text() == "Second"
        assert result["changes"] == ["reject", "approve"]


@pytest.mark.parametrize("kind", ["file", "quote"])
@pytest.mark.parametrize("supplied", [False, True])
def test_workflow_persistence_is_caller_owned(tmp_path, kind, supplied):
    """Factories neither allocate savers nor replace caller-provided persistence."""
    from samples.file_approval.sample import make_simulated_model as file_model
    from samples.quote_request.sample import make_simulated_model as quote_model

    saver = InMemorySaver() if supplied else None
    kwargs = {"checkpointer": saver} if supplied else {}
    if kind == "file":
        source, target, inputs = files(tmp_path)
        graph = file_workflow(
            file_model(inputs["additions"]), source=source, target=target, **kwargs
        )
    else:
        graph = quote_workflow(quote_model(), **kwargs)
    assert graph.checkpointer is saver


@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("read_file", {"file_path": "/outside.txt"}),
        ("write_file", {"file_path": "/source.txt", "content": "Forbidden"}),
        (
            "edit_file",
            {
                "file_path": "/source.txt",
                "old_string": "Original",
                "new_string": "Forbidden",
            },
        ),
        ("write_file", {"file_path": "/../outside.txt", "content": "Forbidden"}),
        (
            "task",
            {"description": "Overwrite source", "subagent_type": "general-purpose"},
        ),
    ],
)
def test_native_file_tools_cannot_escape_editor_scope(tmp_path, tool_name, args):
    """Even autoapproval cannot grant arbitrary paths or delegate around policy."""
    from langchain_core.messages import AIMessage, ToolMessage

    from agent_runtime.harness.simulated_model import ScriptedChatModel

    source, target, _ = files(tmp_path)
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": "denied"}],
            ),
            AIMessage(content="Finished"),
        ]
    )
    graph = file_workflow(model, source=source, target=target, mode="autoapprove")
    result = graph.invoke(request())
    observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert observation.status == "error"
    assert source.read_text() == "Original\n"
    assert not target.exists()
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_edit_rejects_target_changed_during_approval(tmp_path, asynchronous):
    """Both drivers protect native replacements as well as new-file writes."""
    import asyncio

    source, target, inputs = files(tmp_path)
    target.write_text("Original\n")
    graph = editor(inputs)

    async def run():
        first = await graph.ainvoke(request(), CONFIG)
        assert first["__interrupt__"][0].value["tool"] == "edit_file"
        target.write_text("Human edit\n")
        with pytest.raises(RuntimeError, match="Target changed"):
            await graph.ainvoke(Command(resume="approve"), CONFIG)

    if asynchronous:
        asyncio.run(run())
    else:
        first = graph.invoke(request(), CONFIG)
        assert first["__interrupt__"][0].value["tool"] == "edit_file"
        target.write_text("Human edit\n")
        with pytest.raises(RuntimeError, match="Target changed"):
            graph.invoke(Command(resume="approve"), CONFIG)
    assert target.read_text() == "Human edit\n"
    assert source.read_text() == "Original\n"


def test_native_normalized_read_refreshes_target_snapshot(tmp_path):
    """Native path aliases must refresh the guard after an earlier successful write."""
    from langchain_core.messages import AIMessage

    from agent_runtime.harness.simulated_model import ScriptedChatModel

    source, target, _ = files(tmp_path)
    calls = [
        ("write_file", {"file_path": "/target.txt", "content": "First"}),
        ("read_file", {"file_path": "./target.txt"}),
        (
            "edit_file",
            {"file_path": "/target.txt", "old_string": "First", "new_string": "Second"},
        ),
    ]
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": str(index)}]
            )
            for index, (name, args) in enumerate(calls)
        ]
        + [AIMessage(content="Finished")]
    )
    graph = file_workflow(model, source=source, target=target, mode="autoapprove")
    result = graph.invoke(request())
    assert result["status"] == "completed"
    assert target.read_text() == "Second"
