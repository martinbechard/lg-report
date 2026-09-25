"""Dispatch an assignment from a parent graph to a child author/judge review loop.

The parent prepares work and returns the child's result. The child owns drafting,
review, revision, and its round limit. This sample reuses the review roles, while
keeping its own workflow definitions so the two levels can be read together.
Reviews are model assessments; this workflow does not execute generated code.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent_runtime.agents import evidence_judge, review_author, work_planner
from agent_runtime.harness.model_factory import build_model


class ReviewState(TypedDict, total=False):
    """Hold one assignment and the working state of its bounded review loop."""

    assignment: str  # Parent's complete work request.
    author_history: list[BaseMessage]  # Drafts and feedback retained for revision.
    judge_history: list[BaseMessage]  # Judge's prior assessments of this assignment.
    draft: str  # Latest proposed deliverable.
    review: dict  # Validated verdict and feedback from the judge.
    round: int  # Completed drafts, including the first attempt.
    outcome: str  # approved or limit_reached when the child returns.
    answer: str  # Deliverable with unresolved feedback if the limit was reached.


class DeliveryState(TypedDict, total=False):
    """Keep the user conversation and the child's returned result in the parent."""

    messages: Annotated[list[BaseMessage], add_messages]
    assignment: str  # Replaced by the planner for each user turn.
    answer: str  # Returned by the child; never rewritten as an approval.
    outcome: str  # Child review outcome, propagated to the caller.


def _build_review_workflow(model, max_rounds):
    """Compose the same author/judge roles as review_loop in a local child graph."""
    author = review_author.build_agent({"model": model})
    judge = evidence_judge.build_agent({"model": model})

    def begin(state):
        """Start each dispatched assignment with fresh draft and review state."""
        return {
            "author_history": [],
            "judge_history": [],
            "round": 0,
            "draft": "",
            "review": {},
        }

    def write(state, config):
        """Write the assignment or revise it using the previous judge feedback."""
        result = author.invoke(
            {
                "conversation": [HumanMessage(content=state["assignment"])],
                "history": state["author_history"],
                "review": state["review"] if state["round"] else None,
                "high_level": False,
            },
            config=config,
        )
        return {
            "author_history": result["history"],
            "draft": result["draft"],
            "round": state["round"] + 1,
        }

    def assess(state, config):
        """Review every draft against the parent's assignment before returning."""
        result = judge.invoke(
            {
                "conversation": [HumanMessage(content=state["assignment"])],
                "history": state["judge_history"],
                "draft": state["draft"],
            },
            config=config,
        )
        return {"judge_history": result["history"], "review": result["review"]}

    def route(state):
        """Repeat only while the judge requests changes and attempts remain."""
        if state["review"]["verdict"] == "approve" or state["round"] >= max_rounds:
            return "finish"
        return "author"

    def finish(state):
        """Return the draft without disguising an exhausted limit as approval."""
        approved = state["review"]["verdict"] == "approve"
        answer = state["draft"]
        if not approved:
            answer = (
                "Review limit reached — draft NOT approved.\n\n"
                + answer
                + "\n\nOutstanding review:\n"
                + "\n".join(state["review"]["feedback"])
            )
        return {
            "answer": answer,
            "outcome": "approved" if approved else "limit_reached",
        }

    graph = StateGraph(ReviewState)
    graph.add_node("begin", begin, metadata={
        "report_comment": "Reset histories and review rounds",
    })
    # These adapters invoke role models internally. Declare that responsibility
    # so the diagram need not infer an LLM call from a function or role name.
    graph.add_node("author", write, metadata={
        "report_kind": "model",
        "report_comment": "Draft or revise using judge feedback",
    })
    graph.add_node("judge", assess, metadata={
        "report_kind": "model",
        "report_comment": "Assess draft; approve or request fixes",
    })
    graph.add_node("finish", finish, metadata={
        "report_comment": "Return draft and review outcome",
    })
    graph.add_edge(START, "begin")
    graph.add_edge("begin", "author")
    graph.add_edge("author", "judge")
    graph.add_conditional_edges(
        "judge", route, {"author": "author", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    child = graph.compile(name="assignment_review", checkpointer=False)
    child.report_title = "Child review loop"
    child.report_context = (
        f"Separate author and judge histories; at most {max_rounds} drafts. "
        "Review assesses source; no tests run."
    )
    return child


def build_workflow(model=None, *, max_rounds: int = 3):
    """Build parent dispatch and child review using one shared model adapter.

    Graph edges enforce review; no model must echo a precomputed routing action.
    A single ordinary child call works with both sync invocation and LangGraph's
    async execution of synchronous nodes, just like the review_loop sample.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    if model is None:
        model = build_model(caller="workflow")
    planner = work_planner.build_agent({"model": model})
    planner.report_model_comment = "Prepare the complete work assignment"
    child = _build_review_workflow(model, max_rounds)

    def plan(state, config):
        """Let the planner prepare the work that the next node will dispatch."""
        result = planner.invoke({"messages": state["messages"]}, config=config)
        return {"assignment": result["messages"][-1].text}

    def dispatch(state, config):
        """Call the actual compiled child graph and collect its completed result."""
        result = child.invoke({"assignment": state["assignment"]}, config=config)
        return {"answer": result["answer"], "outcome": result["outcome"]}

    def finish(state):
        """Publish the child result unchanged, including any unresolved review."""
        return {"messages": [AIMessage(content=state["answer"])]}

    graph = StateGraph(DeliveryState)
    graph.add_node("planner", plan, metadata={
        "report_comment": "Prepare assignment from user request",
    })
    graph.add_node("review_workflow", dispatch, metadata={
        "report_comment": "Dispatch assignment; await child result",
    })
    graph.add_node("finish", finish, metadata={
        "report_comment": "Publish child result unchanged",
    })
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "review_workflow")
    graph.add_edge("review_workflow", "finish")
    graph.add_edge("finish", END)
    compiled = graph.compile(name="nested_workflows")
    compiled.report_title = "Parent dispatch workflow"
    compiled.report_context = "Plan the assignment, run child review, return its outcome"
    # The adapter changes the state shape, so expose its compiled child for the
    # shared report recorder to discover actual topology rather than copied edges.
    compiled.report_subgraphs = {"review_workflow": child}
    return compiled
