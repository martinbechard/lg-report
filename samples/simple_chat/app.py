"""Lesson 1: a two-turn conversation with no intentional tool use.

The aim is to see system instructions, user messages, and assistant responses
accumulate across calls. Start here before studying tool loops or reasoning.
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.sample_runtime import execute, select_model, settings_for
from samples.simple_chat.simulation import make_simulated_model

SYSTEM_PROMPT = (
    "Answer the user's chat question directly and concisely. "
    "Do not use tools for this simple chat exercise."
)
USER_PROMPTS = [
    "Explain the main steps in an agent workflow.",
    "How does a tool observation help the agent answer?",
]


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    # DeepAgents constructs a real LangGraph graph, even with the offline model.
    # Its built-in tool definitions still occupy input context; that overhead is
    # deliberately visible, despite this lesson making no tool calls.
    # StateBackend keeps any built-in file operations in graph state, not on disk.
    return create_deep_agent(
        model=model,
        backend=StateBackend(),
        subagents=[],
        name="chat-agent",
        system_prompt=SYSTEM_PROMPT,
    )


def main() -> None:
    settings = settings_for(__file__, __doc__)
    model, provider, model_name = select_model(settings, make_simulated_model)
    graph = build_agent(model)
    # Recording is an outer concern: the graph can also be invoked without it.
    execute(
        graph,
        USER_PROMPTS,
        settings,
        provider=provider,
        model=model_name,
        title="Simple chat · two turns",
    )


if __name__ == "__main__":
    main()
