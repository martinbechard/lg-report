"""Nest delivery, coding, and isolated review without implicitly sharing context.

The outer planner/tester history and inner supervisor/coder history are separate
message lists. A wrapper projects an assignment into the coding graph and a
result back out. An opaque-to-the-LLMs coding-memory field preserves the inner
history during test-driven re-entry. Every reviewer invocation starts empty.

This is model-context isolation, not a process or security sandbox. The host and
its trace recorder can inspect every scope. The nested graphs disable their own
checkpoints: explicit state carry-over, not hidden session memory, is the lesson.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent_runtime.agents.nested_development import build_agent
from agent_runtime.harness.model_factory import build_model
from agent_runtime.workflows.nested_contracts import (
    Implementation, Limits, bind_assessment, planner_action,
    require_action, supervisor_action,
)


class DeliveryState(TypedDict, total=False):
    """Outer state; only messages and explicit packets are supplied to its agents."""

    messages: Annotated[list[BaseMessage], add_messages]  # Planner + tester context.
    task_id: str  # Runtime identifier, not an agent-selected session identifier.
    task: dict | None  # Stable, explicit requirements created on the first plan.
    coding_memory: dict  # Host-owned carry-over; NEVER injected into outer prompts.
    coding_result: dict | None  # Deliberate public handoff, not a child transcript.
    test_result: dict | None  # Verdict for the current candidate only.
    coding_cycles: int  # Monotonic across all test-fix re-entry for this task.
    next_action: str  # Validated planner action, checked against deterministic policy.
    planner_note: str
    outcome: str  # pending, completed, or blocked.


class CodingState(TypedDict, total=False):
    """Private child state; supervisor and coder consume the same coding_history."""

    task_id: str
    task: dict
    defects: list[str]  # Explicit tester feedback, not the outer transcript.
    memory: dict  # Previous coding visit's retained history/artifact/attempt count.
    cycle: int
    coding_history: list[BaseMessage]  # Replaced with each peer's complete history.
    artifact: dict | None
    review: dict | None  # Reset on re-entry so an old pass cannot skip development.
    rounds: int  # Attempts in THIS coding visit.
    total_rounds: int  # Attempts across all visits, used for unique candidate IDs.
    directive: str
    next_action: str
    result: dict  # Public output contract assembled only at the return boundary.


class ReviewState(TypedDict, total=False):
    """Third scope: its input is a review packet, never CodingState or DeliveryState."""

    packet: dict
    history_id: str  # Fresh logical context identifier for this individual review.
    assessment: dict


def _scope(config, history_id: str, depth: int):
    """Preserve callbacks while explicitly overriding inherited context labels."""
    return {
        **config,
        "metadata": {
            **config.get("metadata", {}),
            "report_history_id": history_id,
            "report_context_depth": depth,
        },
    }


def _role_node(agent, prepare, accept, identity, depth):
    """Invoke one role on a selected projection, identically in both client modes.

    prepare is intentionally the only place choosing LLM-visible state. accept
    publishes the result into the owning graph; it never receives a model client.
    """
    def run(state, config):
        result = agent.invoke(prepare(state), config=_scope(config, identity(state), depth))
        return accept(state, result)

    async def arun(state, config):
        result = await agent.ainvoke(
            prepare(state), config=_scope(config, identity(state), depth),
        )
        return accept(state, result)

    return RunnableLambda(run, afunc=arun)


def _review_workflow(reviewer):
    """Make the third-level boundary explicit and stateless on every invocation."""
    def prepare(state):
        # No parent history, prior reviewer history, coder notes, or supervisor
        # directives are included. Relevant requirements travel in the packet.
        return {"history": [], "packet": state["packet"]}

    def accept(state, result):
        decision = result["data"]
        bind_assessment(decision, state["packet"]["artifact"])
        # Return only the decision. The fresh reviewer transcript stays private
        # to this invocation and is visible only through ordinary tracing.
        return {"assessment": decision}

    graph = StateGraph(ReviewState)
    graph.add_node("reviewer", _role_node(
        reviewer, prepare, accept, lambda s: s["history_id"], 3,
    ))
    graph.add_edge(START, "reviewer")
    graph.add_edge("reviewer", END)
    return graph.compile(name="isolated_review_workflow", checkpointer=False)


def _coding_workflow(supervisor, coder, review_workflow, limits):
    """Run supervisor -> coder -> isolated review -> supervisor under a hard guard."""
    def begin(state):
        memory = state.get("memory", {})
        return {
            "coding_history": list(memory.get("history", [])),
            "artifact": memory.get("artifact"),
            "total_rounds": memory.get("total_rounds", 0),
            "rounds": 0, "review": None, "next_action": "code", "directive": "",
        }

    def supervisor_input(state):
        allowed = supervisor_action(state["review"], state["rounds"], limits.max_review_rounds)
        return {"history": state["coding_history"], "packet": {
            "allowed_action": allowed,
            "task": state["task"], "cycle": state["cycle"],
            "completed_rounds": state["rounds"],
            "max_review_rounds": limits.max_review_rounds,
            "tester_defects": state["defects"],
            "latest_review": state["review"],
        }}

    def supervisor_output(state, result):
        data = result["data"]
        require_action(data["action"], supervisor_action(
            state["review"], state["rounds"], limits.max_review_rounds,
        ))
        return {"coding_history": result["history"],
                "next_action": data["action"], "directive": data["directive"]}

    def coder_input(state):
        return {"history": state["coding_history"], "packet": {
            "task": state["task"], "directive": state["directive"],
            "tester_defects": state["defects"], "latest_review": state["review"],
            "candidate_id": f"candidate-{state['total_rounds'] + 1}",
        }}

    def coder_output(state, result):
        implementation = Implementation.model_validate(result["data"])
        expected_id = f"candidate-{state['total_rounds'] + 1}"
        if implementation.candidate_id != expected_id:
            raise ValueError("Coder changed the workflow-assigned candidate ID")
        return {
            "coding_history": result["history"],
            "artifact": implementation.handoff().model_dump(),
            "review": None,  # Any earlier review is now invalid.
            "rounds": state["rounds"] + 1,
            "total_rounds": state["total_rounds"] + 1,
        }

    def review_input(state):
        # This projection is the context boundary. Never pass the entire state
        # or copy coding_history into a freshly named reviewer session.
        return {
            "history_id": f"review:{state['task_id']}:{state['total_rounds']}",
            "packet": {"task": state["task"], "artifact": state["artifact"],
                       "reported_defects": state["defects"]},
        }

    def review(state, config):
        result = review_workflow.invoke(review_input(state), config=config)
        return {"review": result["assessment"]}

    async def areview(state, config):
        result = await review_workflow.ainvoke(review_input(state), config=config)
        return {"review": result["assessment"]}

    def finish(state):
        passed = state["next_action"] == "return"
        if passed:
            bind_assessment(state["review"], state["artifact"])
            if state["review"]["verdict"] != "pass":
                raise ValueError("Cannot return unapproved code for testing")
        return {"result": {
            "status": "approved_for_testing" if passed else "blocked",
            "artifact": state["artifact"], "review": state["review"],
            "rounds_this_cycle": state["rounds"],
            "total_rounds": state["total_rounds"],
            "reason": "Review passed" if passed else "Review-round limit reached",
        }}

    def identity(state):
        """Keep the coding context label stable across visits for this task."""
        return f"coding:{state['task_id']}"
    graph = StateGraph(CodingState)
    graph.add_node("begin_coding_visit", begin)
    graph.add_node("coding_supervisor", _role_node(
        supervisor, supervisor_input, supervisor_output, identity, 2,
    ))
    graph.add_node("coder", _role_node(coder, coder_input, coder_output, identity, 2))
    graph.add_node("isolated_review", RunnableLambda(review, afunc=areview))
    graph.add_node("return_coding_result", finish)
    graph.add_edge(START, "begin_coding_visit")
    graph.add_edge("begin_coding_visit", "coding_supervisor")
    graph.add_conditional_edges("coding_supervisor", lambda s: s["next_action"], {
        "code": "coder", "return": "return_coding_result",
        "escalate": "return_coding_result",
    })
    graph.add_edge("coder", "isolated_review")
    graph.add_edge("isolated_review", "coding_supervisor")
    graph.add_edge("return_coding_result", END)
    return graph.compile(name="coding_workflow", checkpointer=False)


def build_workflow(planner_model=None, supervisor_model=None, coder_model=None,
                   reviewer_model=None, tester_model=None, *,
                   max_review_rounds: int = 3, max_coding_cycles: int = 3,
                   scenario: str = "rework"):
    """Build the new sample using the existing model factory and sample catalog.

    No provider call occurs during construction. Defaults keep both graphs well
    below the standard recursion guard. Raising limits substantially may require
    a larger invocation recursion_limit; that framework guard is only a backup.
    Errors or invalid model JSON fail visibly instead of being retried forever.
    """
    # SampleCatalog passes the same options to fixture and workflow factories.
    # The scenario selects OFFLINE scripts only; live verdicts are not forced.
    if scenario not in {"rework", "review_limit", "test_limit", "first_pass"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    limits = Limits(max_review_rounds, max_coding_cycles)
    supplied = dict(zip(
        ("planner", "coding_supervisor", "coder", "reviewer", "tester"),
        (planner_model, supervisor_model, coder_model, reviewer_model, tester_model),
        strict=True,
    ))
    # Workflow ownership is explicit, just as in the original sample. Agents
    # own role instructions; the workflow owns tools, middleware and retention.
    agents = {
        name: build_agent(name, {
            "model": model if model is not None else build_model(caller=name),
            "tools": [], "middleware": [], "checkpointer": False,
        })
        for name, model in supplied.items()
    }
    review_workflow = _review_workflow(agents["reviewer"])
    coding_workflow = _coding_workflow(
        agents["coding_supervisor"], agents["coder"], review_workflow, limits,
    )

    def begin(state):
        # Reset private work for a NEW user task, not for a test-fix re-entry.
        # The outer conversation may be retained by the existing chat harness.
        return {
            "messages": convert_to_messages(state.get("messages", [])),
            "task_id": str(uuid4()), "task": None, "coding_memory": {},
            "coding_result": None, "test_result": None, "coding_cycles": 0,
            "next_action": "code", "planner_note": "", "outcome": "pending",
        }

    def planner_input(state):
        return {"history": state["messages"], "packet": {
            "allowed_action": planner_action(
                state["coding_result"], state["test_result"],
                state["coding_cycles"], limits.max_coding_cycles,
            ),
            "stable_task": state["task"],
            "coding_cycles_used": state["coding_cycles"],
            "max_coding_cycles": limits.max_coding_cycles,
        }}

    def planner_output(state, result):
        data = result["data"]
        require_action(data["action"], planner_action(
            state["coding_result"], state["test_result"],
            state["coding_cycles"], limits.max_coding_cycles,
        ))
        if state["task"] is not None and data["task"] != state["task"]:
            raise ValueError("A re-entry cannot silently change the acceptance contract")
        # No history is removed or summarized. Append just the new request/reply
        # so add_messages does not duplicate the previous shared conversation.
        return {"messages": result["history"][len(state["messages"]):],
                "task": data["task"], "next_action": data["action"],
                "planner_note": data["note"]}

    def coding_input(state):
        if state["coding_cycles"] >= limits.max_coding_cycles:
            raise ValueError("Coding-cycle guard forbids another child invocation")
        return {
            "task_id": state["task_id"], "task": state["task"],
            "defects": [] if state["test_result"] is None else state["test_result"]["findings"],
            "memory": state["coding_memory"], "cycle": state["coding_cycles"] + 1,
        }

    def coding_output(state, child):
        result = child["result"]
        # Public handoff and private carry-over are intentionally distinct.
        # The planner/tester sees result, but never the memory history below.
        return {
            "coding_memory": {"history": child["coding_history"],
                              "artifact": child["artifact"],
                              "total_rounds": child["total_rounds"]},
            "coding_result": result, "test_result": None,
            "coding_cycles": state["coding_cycles"] + 1,
            "messages": [HumanMessage(name="coding_workflow_result", content=json.dumps(result))],
        }

    def coding(state, config):
        # Actual nested workflow invocation, not five flattened peer agents.
        child = coding_workflow.invoke(coding_input(state), config=config)
        return coding_output(state, child)

    async def acoding(state, config):
        child = await coding_workflow.ainvoke(coding_input(state), config=config)
        return coding_output(state, child)

    def tester_input(state):
        if state["coding_result"]["status"] != "approved_for_testing":
            raise ValueError("Tester must not receive blocked coding work")
        return {"history": state["messages"], "packet": {
            "task": state["task"], "artifact": state["coding_result"]["artifact"],
        }}

    def tester_output(state, result):
        bind_assessment(result["data"], state["coding_result"]["artifact"])
        return {"messages": result["history"][len(state["messages"]):],
                "test_result": result["data"]}

    def finish(state):
        completed = state["next_action"] == "finish"
        status = "completed" if completed else "blocked"
        if completed:
            explanation = (
                "The scripted/model assessment passed. This sample did not execute code or tests."
            )
        elif state["coding_result"] and state["coding_result"]["status"] == "blocked":
            explanation = "Review-round limit reached. The candidate is NOT approved for testing."
        else:
            explanation = "Test-fix cycle limit reached. Acceptance defects remain unresolved."
        return {"outcome": status, "messages": [AIMessage(
            name="planner", content=f"{status.upper()}: {state['planner_note']}\n{explanation}",
        )]}

    def identity(state):
        """Identify the one history shared by the planner and tester."""
        return f"outer:{state['task_id']}"
    graph = StateGraph(DeliveryState)
    graph.add_node("begin_delivery", begin)
    graph.add_node("planner", _role_node(
        agents["planner"], planner_input, planner_output, identity, 1,
    ))
    graph.add_node("coding_workflow", RunnableLambda(coding, afunc=acoding), metadata={
        "report_description": "Enter a nested workflow with a separate shared coding context.",
    })
    graph.add_node("tester", _role_node(
        agents["tester"], tester_input, tester_output, identity, 1,
    ))
    graph.add_node("finish_delivery", finish)
    graph.add_edge(START, "begin_delivery")
    graph.add_edge("begin_delivery", "planner")
    graph.add_conditional_edges("planner", lambda s: s["next_action"], {
        "code": "coding_workflow", "test": "tester",
        "finish": "finish_delivery", "escalate": "finish_delivery",
    })
    graph.add_edge("coding_workflow", "planner")
    graph.add_edge("tester", "planner")
    graph.add_edge("finish_delivery", END)
    return graph.compile(name="nested_workflows")
