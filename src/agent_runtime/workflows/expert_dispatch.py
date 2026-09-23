"""Connect a dispatcher to movie, sports, and history expert graphs.

The workflow requests one shared LLM from build_model; callers may inject one
for focused tests. Domain data and user questions do not enter construction.
This workflow gives that LLM to every agent and registers the experts with the
dispatcher. Role instructions remain in the individual agent files.
See docs/chat-composition.md for the composition and delegation diagrams.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.backends import StateBackend
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agent_runtime.agents import dispatcher_agent, reference_expert
from agent_runtime.harness.model_factory import build_model


def build_workflow(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Give the application a dispatcher that can consult three domain experts.

    Return the compiled dispatcher graph with movie, sports, and history roles
    available through its task tool. The dispatcher chooses delegation during
    execution; building the expert graphs does not ask them a question.

    model is the single configured provider adapter or offline simulator. Each
    agent combines it with its own instructions, tools, and graph-owned message
    history. Sharing the LLM does not merge those conversations.
    Construction makes no model requests; Conversation invokes the returned graph.
    """
    # Agent delegation flow (middleware owns task routing and return edges):
    #
    # input messages -> dispatcher -> final answer
    #                      |  ^
    #     task(assignment) |  | selected expert's final answer as tool result
    #                      v  |
    #              +-------+--+-------+
    #              |       |          |
    #              v       v          v
    #       movie_expert sports_expert history_expert
    #              | ^     | ^        | ^
    #              v |     v |        v |
    #         movie lookup sports lookup history lookup
    #
    # The three branches are available choices, not a mandatory fan-out or a
    # sequence. Each selected expert returns to the same dispatcher, which may
    # delegate again or answer. Each lookup is that expert's own reference tool;
    # model-selected tool calls return evidence to the calling expert's loop.
    # These are nested agent/tool calls, not custom outer StateGraph nodes.
    # All roles share one model adapter but retain separate message histories.
    # Model selection belongs to the factory; the workflow supplies its role.
    if model is None:
        model = build_model(caller="workflow")
    # Expertise comes from the role's instructions and retrieval tool. All four
    # agents use this same LLM object; there are no per-expert model settings.
    parameters = {"model": model}
    # Domain definitions own prompts/tools. The workflow registers participants
    # without duplicating their implementation or merging their graph state.
    experts: list[CompiledSubAgent] = [
        {
            "name": definition.name,
            "description": definition.description,
            "runnable": reference_expert.build_agent(parameters, definition),
        }
        for definition in (
            reference_expert.MOVIE,
            reference_expert.SPORTS,
            reference_expert.HISTORY,
        )
    ]
    backend = StateBackend()
    # LangChain receives its own public arguments. The DeepAgents delegation
    # middleware carries the backend it needs; create_agent has no backend keyword.
    dispatcher_parameters = {
        **parameters,
        "middleware": [
            SubAgentMiddleware(
                backend=backend,
                subagents=experts,
                task_description="Delegate a self-contained question to the matching expert. Available experts:\n{available_agents}",
            ),
        ],
    }
    return dispatcher_agent.build_agent(dispatcher_parameters)
