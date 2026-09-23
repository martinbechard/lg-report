"""Share compacted history between peers while budgeting an isolated child.

The workflow owns messages and context budgets; agent modules own role prompts
and tools. Planner then responder consume the same evolving history. The task
tool starts a specialist with its assignment only and returns its final answer.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.backends import StateBackend
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent_runtime.agents import context_planner, context_responder, workflow_specialist
from agent_runtime.context_budget import ContextBudget
from agent_runtime.harness.model_factory import build_model

# Small teaching budgets make compaction visible without a long conversation.
# These are input estimates, not changes to the provider's physical capacity.
WORKFLOW_BUDGET = ContextBudget(
    max_input_tokens=4000, trigger_tokens=500, keep_tokens=120
)
SUBAGENT_BUDGET = ContextBudget(
    max_input_tokens=2000, trigger_tokens=180, keep_tokens=60
)


def _shared_history_node(agent):
    """Publish the peer's retained history, including middleware deletions.

    Returning only remaining messages would let the outer add_messages reducer
    resurrect old messages deleted inside the peer. Replace the outer list with
    the peer's complete result, so the next peer and next turn see its summary.
    Both paths inherit callbacks/config, including the owning graph checkpointer.
    """

    def run(state, config):
        result = agent.invoke({"messages": state["messages"]}, config)
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *result["messages"]]
        }

    async def arun(state, config):
        result = await agent.ainvoke({"messages": state["messages"]}, config)
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *result["messages"]]
        }

    return RunnableLambda(run, afunc=arun)


def build_workflow(
    planner_model=None,
    responder_model=None,
    specialist_model=None,
    workflow_summary_model=None,
    subagent_summary_model=None,
    *,
    workflow_budget: ContextBudget = WORKFLOW_BUDGET,
    subagent_budget: ContextBudget = SUBAGENT_BUDGET,
):
    """Compile peer sharing and native task delegation without invoking a model.

    The two budget arguments are the workflow-level configuration surface.
    Fresh middleware instances use the same peer policy; no mutable policy state
    or parent history is handed to the isolated child. Summary models are explicit
    dependencies, and summarization calls remain visible in normal tracing.
    The harness supplies persistence at this outer graph's execution boundary.
    """
    # Each role requests its own model; scripted ledgers remain independent.
    if planner_model is None:
        planner_model = build_model(caller="planner")
    if responder_model is None:
        responder_model = build_model(caller="responder")
    if specialist_model is None:
        specialist_model = build_model(caller="workflow-specialist")
    if workflow_summary_model is None:
        workflow_summary_model = build_model(caller="workflow-summary")
    if subagent_summary_model is None:
        subagent_summary_model = build_model(caller="subagent-summary")
    specialist = workflow_specialist.build_agent(
        {
            "model": specialist_model,
            "middleware": subagent_budget.middleware(subagent_summary_model),
        }
    )
    # This is the answer to configuring middleware "with the subagent tool":
    # register it on the specification, not in model-generated task arguments.
    specialist["mode"] = "isolated"
    planner = context_planner.build_agent(
        {
            "model": planner_model,
            "middleware": workflow_budget.middleware(workflow_summary_model),
        }
    )
    responder = context_responder.build_agent(
        {
            "model": responder_model,
            "middleware": (
                *workflow_budget.middleware(workflow_summary_model),
                SubAgentMiddleware(backend=StateBackend(), subagents=[specialist]),
            ),
        }
    )
    graph = StateGraph(MessagesState)
    graph.add_node("planning_peer", _shared_history_node(planner))
    graph.add_node("responding_peer", _shared_history_node(responder))
    graph.add_edge(START, "planning_peer")
    graph.add_edge("planning_peer", "responding_peer")
    graph.add_edge("responding_peer", END)
    return graph.compile(name="context_budget_workflow")
