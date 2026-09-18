"""Lesson 2: connect a model decision, local tool execution, and follow-up request.

The model produces tool-call JSON; LangGraph executes the named function and
appends its observation. The next model call consumes that expanded history.
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.sample_runtime import execute, select_model, settings_for
from samples.tool_chat.simulation import make_simulated_model
from samples.tool_chat.tools import workflow_reference

SYSTEM_PROMPT = "Use workflow_reference to obtain evidence for each question about agent workflows, then answer concisely using the observation."
USER_PROMPTS = [
    "Explain the main steps in an agent workflow.",
    "How does a tool observation help the agent answer?",
]


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    # Register the actual function, not a fabricated tool-result string. This is
    # what makes the resulting trace a runnable tool-use example rather than a
    # preassembled report. LangGraph owns the model/tool/model routing.
    return create_deep_agent(
        model=model,
        tools=[workflow_reference],
        backend=StateBackend(),
        name="reference-chat-agent",
        system_prompt=SYSTEM_PROMPT,
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
        title="Reference lookup · two turns",
    )


if __name__ == "__main__":
    main()
