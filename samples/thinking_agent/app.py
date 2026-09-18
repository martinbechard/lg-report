"""Lesson 3: evidence collection, expensive reasoning, then verification.

Use this application to distinguish visible output, reasoning-token charges,
and tool-result input. Tool evidence and checks describe a fictional service;
they do not inspect or load-test the developer's computer.
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.sample_runtime import execute, select_model, settings_for
from samples.thinking_agent.simulation import make_simulated_model
from samples.thinking_agent.tools import inspect_service, test_plan

SYSTEM_PROMPT = "Use the fictional service evidence to propose and validate a latency improvement. Inspect traffic, database behavior, and constraints before testing the plan."
USER_PROMPTS = [
    "Investigate service latency and validate a plan meeting the constraints."
]


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    # These tools have separate responsibilities: one supplies evidence; the
    # other supplies fixture results for evaluating a candidate plan. The model
    # decides when to call each through the normal LangGraph tool loop.
    # In offline mode the sequence is deterministic. A live model may choose a
    # different number/order of calls, and its reported token usage is retained.
    return create_deep_agent(
        model=model,
        tools=[inspect_service, test_plan],
        backend=StateBackend(),
        system_prompt=SYSTEM_PROMPT,
        name="investigation-agent",
    )


def main() -> None:
    settings = settings_for(__file__, __doc__)
    model, provider, model_name = select_model(settings, make_simulated_model)
    graph = build_agent(model)
    execute(
        graph,
        USER_PROMPTS,
        settings,
        provider=provider,
        model=model_name,
        title="Investigation · reasoning and tools",
    )


if __name__ == "__main__":
    main()
