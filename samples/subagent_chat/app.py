"""Teach isolated parent/subagent delegation with a local HTML execution report.

The parent delegates through DeepAgents' real task tool. The specialist owns
its lookup tool and returns only its final answer to the parent. No application
code manually invokes the specialist or fabricates the tool results.
AI attribution: Generated with AI assistance.
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.model_config import configured_model
from lg_report.sample_runtime import execute, settings_for
from samples.subagent_chat.simulation import make_simulated_models
from samples.tool_chat.tools import workflow_reference

# The user question is deliberately broader than the specialist's bounded task.
USER_PROMPTS = [
    "Explain ReAct and how a specialist's tool observations help a parent agent answer."
]
PARENT_PROMPT = "Delegate the workflow lookup to workflow-specialist using task. Provide a self-contained assignment. After its summary returns, answer the user concisely. Do not perform the lookup yourself."
SPECIALIST_PROMPT = "You are a workflow reference specialist. Call workflow_reference for evidence about the assigned topic, then return a concise factual summary to the parent. Do not delegate further."


def build_agent(parent: BaseChatModel, specialist: BaseChatModel) -> CompiledStateGraph:
    """Compile the parent and isolated specialist around the supplied models.

    The specialist gets its own instructions and tools. Default isolated mode
    starts it with the delegated task, not the parent's entire conversation.
    StateBackend keeps framework file operations in memory. Graph construction
    performs no model or network calls; callbacks attach later at invocation.
    """
    return create_deep_agent(
        model=parent,
        backend=StateBackend(),
        name="delegating-parent",
        system_prompt=PARENT_PROMPT,
        subagents=[
            {
                "name": "workflow-specialist",
                "description": "Looks up agent workflow concepts and returns an evidence-based summary.",
                "system_prompt": SPECIALIST_PROMPT,
                "model": specialist,
                "tools": [workflow_reference],
            }
        ],
    )


def create_graph(live: bool) -> CompiledStateGraph:
    """Build a fresh run; live mode uses two adapters with the same .env configuration.

    Each agent gets its own model object. In simulation this is essential: the
    context ledger and scripted response position belong to one agent only.
    Missing live credentials raise before the graph is invoked.
    """
    if live:
        parent = configured_model()[0]
        specialist = configured_model()[0]
    else:
        parent, specialist = make_simulated_models()
    return build_agent(parent, specialist)


def main() -> None:
    """Load this sample's configuration and record the full nested invocation."""
    settings = settings_for(__file__, __doc__)
    # Each live adapter emits its provider/model identity in callback metadata.
    # Explicit configured identity also supplies the recorder's root information.
    if settings.live:
        parent, provider, model_name = configured_model()
        graph = build_agent(parent, configured_model()[0])
    else:
        graph = create_graph(False)
        provider, model_name = "demo", "scripted-chat"
    execute(
        graph,
        USER_PROMPTS,
        settings,
        provider=provider,
        model=model_name,
        title="Parent and specialist · delegation",
    )


if __name__ == "__main__":
    main()
