"""Read a text file and gate each proposed output modification independently.
Approval and writes use separate nodes so resuming a pause cannot repeat a write.
Checkpoints are session-local; the caller supplies explicit source and target paths.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from difflib import unified_diff
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class FileState(TypedDict, total=False):
    """State carried between one proposed file change and the next.

    The workflow intentionally keeps the source text, the target snapshot, the
    proposed replacement, and each user's decision as separate fields.  That
    separation lets the approval pause show exactly what will be written and
    lets the apply node detect a concurrent target edit before doing I/O.
    ``total=False`` reflects the graph lifecycle: ``read`` creates the initial
    fields and later nodes add the review fields incrementally.
    """

    # Path of the file whose text is used as the immutable starting content for
    # this run.  Reading it is an explicit workflow input, not a directory scan.
    source: str
    # Path that receives approved content; it may be a new file.
    target: str
    # ``always-ask`` pauses for each addition; ``autoapprove`` skips the pause.
    mode: str
    # Ordered text fragments.  Each fragment is reviewed and applied separately
    # so a later rejection does not erase an earlier approved change.
    additions: list[str]
    # Index of the fragment currently being prepared or just completed.
    index: int
    # Current accumulated content after the last completed decision.
    content: str
    # Target text as decoded at review time, or None when target was absent.  It
    # is a check-before-write guard, not an atomic filesystem transaction.
    before: str | None
    # Candidate target text shown to the user and guarded by ``before``.
    proposed: str
    # User or mode decision for the current candidate.
    decision: str
    # ``pending`` before a decision, ``completed`` after each decision, or
    # ``cancelled`` after cancellation; index still determines total completion.
    status: str
    # Ordered audit trail containing one decision per proposed fragment.
    changes: list[str]


def build_workflow():
    """Let a caller review each requested file addition before it changes the target.

    Return a compiled graph that accumulates approved additions and records each
    decision. Invoke it with source/target paths, ordered additions, and an
    optional mode; each node returns a partial update merged into FileState.

    The caller supplies paths and additions when invoking the compiled graph;
    construction performs no filesystem access.  ``always-ask`` is the default
    and invalid decisions fail closed.  The checkpointer is process-local, so a
    caller can resume an interrupt in the same process but this function does
    not provide durable approval records across process restarts.
    """

    def read(state):
        """Establish the starting text before reviewing any requested additions.

        At the START edge, ``state`` supplies source, additions, and optional mode.
        Return the source content and fresh decision counters; the next edge goes
        to prepare when additions exist, or END when there is no work.

        The mode check occurs before reading so a typo cannot silently select a
        permissive path.  An empty additions list is already complete; otherwise
        the first fragment begins at index zero.  This node does not touch the
        target because no candidate has been prepared for review yet.
        """
        if state.get("mode", "always-ask") not in {"always-ask", "autoapprove"}:
            raise ValueError("Unknown approval mode")
        content = Path(state["source"]).read_text(encoding="utf-8")
        return {
            "content": content,
            "index": 0,
            "changes": [],
            "status": "pending" if state["additions"] else "completed",
        }

    def prepare(state):
        """Prepare a concrete file change for the reviewer to accept or reject.

        ``state`` supplies the target path, approved content, and current addition
        index. Return ``before`` and ``proposed`` as a partial state update, then
        pass control to approve; this preparation performs reads but no writes.

        A missing target is represented by ``None`` so creation and replacement
        use the same approval protocol.  The candidate is based on the content
        accumulated by prior approved changes, which makes each addition's
        effect explicit and preserves the requested ordering.
        """
        target = Path(state["target"])
        before = target.read_text(encoding="utf-8") if target.exists() else None
        return {
            "before": before,
            "proposed": state["content"] + state["additions"][state["index"]],
        }

    def approve(state):
        """Obtain authorization for this candidate before any target write can occur.

        ``state`` supplies the target snapshot, proposed content, and mode. Return
        only a decision update: recognized choices route to apply; invalid input
        routes to a new approval attempt. Autoapprove explicitly authorizes the
        candidate without calling a human responder.

        The interrupt payload contains the unified diff and full proposed text
        because reviewers need both a compact comparison and an unambiguous
        write value.  Interrupting before ``apply`` is deliberate: resuming the
        approval node can recompute the decision, but cannot repeat a completed
        write.  Unknown resume values become ``invalid`` and route back through
        this node rather than being treated as approval.
        """
        decision = "approve"
        if state.get("mode", "always-ask") == "always-ask":
            diff = "".join(
                unified_diff(
                    (state["before"] or "").splitlines(keepends=True),
                    state["proposed"].splitlines(keepends=True),
                    fromfile=state["target"],
                    tofile=state["target"],
                )
            )
            # First pass: interrupt raises LangGraph's internal GraphInterrupt;
            # this function exits before assigning the human response to decision.
            # LangGraph catches it and returns the payload in __interrupt__ to
            # the invoking client.
            # The client calls this same graph/thread with Command(resume=answer).
            # LangGraph restarts approve from its beginning: the diff is rebuilt,
            # then this interrupt returns that answer and validation below runs.
            # Completed read/prepare nodes are not replayed merely by resuming;
            # apply has not run yet. This is node replay, not a suspended stack.
            decision = interrupt(
                {
                    "kind": "approval",
                    "target": state["target"],
                    "diff": diff,
                    "content": state["proposed"],
                    "choices": ["approve", "reject", "cancel"],
                }
            )
        # An invalid answer finishes this attempt with an invalid marker. Its
        # conditional edge starts a NEW approve attempt and therefore a new pause;
        # it does not keep reusing the prior resume answer indefinitely.
        if not isinstance(decision, str) or decision not in {
            "approve",
            "reject",
            "cancel",
        }:
            return {"decision": "invalid"}
        return {"decision": decision}

    def apply(state):
        """Resolve the reviewed addition so the run can advance or stop on cancellation.

        ``state`` contains a recognized decision and the exact reviewed candidate.
        Write only an approved candidate, then return updated accumulated content,
        decision history, index, and status for next_step to inspect.

        The equality check is a best-effort optimistic-concurrency guard: text
        changed by another actor after review raises an error before writing.
        A caller must start again to arrange review of the changed target.
        The check and subsequent write are not an atomic filesystem transaction,
        so a race after the check remains a caller-visible limitation.
        Rejections do not write, while cancellation
        ends the run after recording the decision.  The returned content is
        therefore the approved accumulation or the unchanged prior accumulation.
        """
        decision = state["decision"]
        if decision == "approve":
            target = Path(state["target"])
            current = target.read_text(encoding="utf-8") if target.exists() else None
            if current != state["before"]:
                raise RuntimeError(
                    "Target changed after review; start again to review the new content"
                )
            target.write_text(state["proposed"], encoding="utf-8")
        # Rejection consumes this fragment without adding it to content. Cancel
        # also preserves earlier writes; neither choice erases checkpoint history.
        return {
            "content": state["proposed"] if decision == "approve" else state["content"],
            "index": state["index"] + 1,
            "changes": [*state["changes"], decision],
            "status": "cancelled" if decision == "cancel" else "completed",
        }

    def next_step(state):
        """Stop after cancellation or the last addition; otherwise review the next one.

        LangGraph supplies ``state`` after apply has incremented the index and
        recorded status. Return END (the framework's terminal sentinel) or the
        node name ``prepare``; this selects execution without changing state.

        Keeping this routing separate makes the terminal conditions visible and
        ensures a rejected fragment can still be followed by a later review.
        """
        if state["status"] == "cancelled" or state["index"] >= len(state["additions"]):
            return END
        return "prepare"

    # The graph keeps review and mutation in different nodes.  This is the key
    # safety property for interrupt/resume: the pause happens before any write,
    # and only the apply node is allowed to mutate the target path.
    # Actual return edges (prepare and approve each denote a single node):
    #
    # START -> read -- no additions ----------------------------------> END
    #            |
    #         additions
    #            v
    #         prepare <--------------------------------------------+
    #            |                                                 |
    #            v                                                 |
    #         approve <---+                                        |
    #            |        |                                        |
    #            +-- invalid                                       |
    #            |                                                 |
    #         approve / reject / cancel                            |
    #            v                                                 |
    #          apply -- cancelled or no additions left ------------> END
    #            |                                                 |
    #            +-- more additions, not cancelled -----------------+
    #
    # In always-ask mode, approve pauses and resumes with the human's decision;
    # autoapprove skips that pause. Only an approve decision writes in apply.
    # A changed target raises an error before writing; it has no recovery edge.
    #
    graph = StateGraph(FileState)
    for name, node in [
        ("read", read),
        ("prepare", prepare),
        ("approve", approve),
        ("apply", apply),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "read")
    # No additions means there is nothing to approve or write, so the run exits
    # immediately after initializing its explicit completed status.
    # Here s is FileState after read's update. The lambda returns a node name
    # or END; evaluating the route performs no file work itself.
    graph.add_conditional_edges("read", lambda s: "prepare" if s["additions"] else END)
    graph.add_edge("prepare", "approve")
    # Invalid resume data loops to approval; only the three accepted decisions
    # can reach apply and therefore reach the filesystem mutation boundary.
    # Here s is FileState after approve stored its decision. Returning approve
    # schedules another attempt; returning apply schedules decision processing.
    graph.add_conditional_edges(
        "approve", lambda s: "approve" if s["decision"] == "invalid" else "apply"
    )
    # Each applied decision increments the fragment index before this edge runs,
    # so prepare always sees the next ordered addition.
    graph.add_conditional_edges("apply", next_step)
    # Keep this compiled graph (and its saver) alive across pauses. Initial and
    # resumed invokes must use the same config["configurable"]["thread_id"];
    # a new graph or different thread cannot retrieve this pending approval.
    return graph.compile(checkpointer=InMemorySaver())
