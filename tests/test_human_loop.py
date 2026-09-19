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
from langgraph.types import Command

from lg_report.workflows.file_approval import build_workflow as file_workflow
from lg_report.workflows.quote_request import build_workflow as quote_workflow

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


# Every write must pause for a decision, reject unknown decisions
# safely, and preserve the source and already approved changes.
def test_each_write_requires_approval(tmp_path):
    source, target, inputs = files(tmp_path)
    graph = file_workflow()
    first = graph.invoke(inputs, CONFIG)
    assert not target.exists()
    assert "First" in first["__interrupt__"][0].value["diff"]
    second = graph.invoke(Command(resume="approve"), CONFIG)
    assert target.read_text() == "Original\nFirst\n"
    assert second["__interrupt__"]
    # Unknown decisions fail closed and ask again, without writing.
    invalid = graph.invoke(Command(resume="yes"), CONFIG)
    assert invalid["__interrupt__"]
    assert target.read_text() == "Original\nFirst\n"
    result = graph.invoke(Command(resume="approve"), CONFIG)
    assert not result.get("__interrupt__")
    assert result["changes"] == ["approve", "approve"]
    assert target.read_text() == "Original\nFirst\nSecond\n"
    assert source.read_text() == "Original\n"


# Auto-approval is an explicit mode; it should complete both writes
# without creating an interrupt while still producing the expected file.
def test_autoapprove_never_pauses(tmp_path):
    _, target, inputs = files(tmp_path, "autoapprove")
    result = file_workflow().invoke(inputs, CONFIG)
    assert not result.get("__interrupt__")
    assert target.read_text() == "Original\nFirst\nSecond\n"


# Rejecting the first diff must remove it from later state so a later
# approval applies only the remaining change.
def test_reject_then_approve_does_not_reintroduce_rejected_change(tmp_path):
    _, target, inputs = files(tmp_path)
    graph = file_workflow()
    graph.invoke(inputs, CONFIG)
    graph.invoke(Command(resume="reject"), CONFIG)
    assert not target.exists()
    result = graph.invoke(Command(resume="approve"), CONFIG)
    assert result["changes"] == ["reject", "approve"]
    assert target.read_text() == "Original\nSecond\n"


@pytest.mark.parametrize("first", [True, False])
# Cancellation must stop future writes whether it occurs before the
# first approval or after one approved change has already been committed.
def test_file_cancel_stops_future_changes(tmp_path, first):
    _, target, inputs = files(tmp_path)
    graph = file_workflow()
    graph.invoke(inputs, CONFIG)
    if not first:
        graph.invoke(Command(resume="approve"), CONFIG)
    result = graph.invoke(Command(resume="cancel"), CONFIG)
    assert result["status"] == "cancelled"
    assert not result.get("__interrupt__")
    assert (not target.exists()) if first else target.read_text() == "Original\nFirst\n"


# Detect a human edit made after review began and fail closed instead
# of overwriting the newer target content.
def test_changed_target_is_not_overwritten(tmp_path):
    _, target, inputs = files(tmp_path)
    graph = file_workflow()
    graph.invoke(inputs, CONFIG)
    target.write_text("Human edit\n")
    with pytest.raises(RuntimeError, match="Target changed"):
        graph.invoke(Command(resume="approve"), CONFIG)
    assert target.read_text() == "Human edit\n"


# Source-as-target edits and existing output still require review,
# proving approval protects in-place changes as well as new files.
def test_in_place_edit_and_existing_output_are_reviewed(tmp_path):
    source, _, inputs = files(tmp_path)
    inputs["target"] = str(source)
    graph = file_workflow()
    first = graph.invoke(inputs, CONFIG)
    assert first["before"] == "Original\n"
    graph.invoke(Command(resume="approve"), CONFIG)
    assert source.read_text() == "Original\nFirst\n"
    graph.invoke(Command(resume="reject"), CONFIG)
    assert source.read_text() == "Original\nFirst\n"


# Field presence alone must not bypass a scripted clarification decision.
# Each human answer must reach the next assessment in the retained context.
def test_quote_model_questions_resolve_multiple_issues():
    """Complete fields still pause; each human answer reaches the next model call."""
    from langchain_core.callbacks import BaseCallbackHandler

    from lg_report.agents.quote_interpreter import SYSTEM_PROMPT
    from samples.quote_request.test_case import (
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
    graph = quote_workflow(make_simulated_model())
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
    assert capture.agent_runs == ["quote_interpreter"] * 4
    assert not result.get("__interrupt__")


# A clear request may complete immediately; the workflow must follow
# the model decision rather than enforce a fixed question count.
def test_quote_model_can_complete_without_questions():
    """Routing follows the model, not mandatory fields or a fixed question count."""
    from samples.quote_request.test_case import make_simulated_model

    decision = {
        "action": "complete",
        "reason": "Scope is clear.",
        "text": "Print 20 identical A4 posters; customer supplies artwork.",
    }
    result = quote_workflow(make_simulated_model([decision])).invoke(
        {"values": {"description": decision["text"]}}, CONFIG
    )
    assert result["status"] == "completed"
    assert not result.get("__interrupt__")


# A vague human answer can leave the request unresolved, so the model
# gets another turn with the first answer retained in conversation state.
def test_quote_unclear_answer_can_trigger_another_question():
    """The human's first reply does not automatically resolve the model's concern."""
    from samples.quote_request.test_case import (
        DECISIONS,
        INITIAL_VALUES,
        make_simulated_model,
    )

    graph = quote_workflow(make_simulated_model([DECISIONS[0], DECISIONS[0]]))
    graph.invoke({"values": INITIAL_VALUES}, CONFIG)
    result = graph.invoke(Command(resume="Whatever is best"), CONFIG)
    assert result["__interrupt__"]
    assert result["request"] is None
    assert result["conversation"][0]["answer"] == "Whatever is best"


# Explicit cancellation must clear partial values, conversation, and request
# from the returned state. This does not test deletion of checkpoint history.
def test_quote_abandon_discards_partial_request():
    from samples.quote_request.test_case import INITIAL_VALUES, make_simulated_model

    graph = quote_workflow(make_simulated_model())
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

    from lg_report.platform.simulated_model import ScriptedChatModel

    model = ScriptedChatModel(responses=[AIMessage(content="Looks fine")])
    with pytest.raises(ValueError, match="exactly one QuoteDecision"):
        quote_workflow(model).invoke({"values": {}}, CONFIG)


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
        f"samples.{sample}.app",
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
        args += ["--source", str(source), "--target", str(target)]
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
