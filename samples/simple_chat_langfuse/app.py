"""Teach Langfuse tracing with a two-turn DeepAgents conversation.

This application owns its graph and invocation boundary. It reuses the baseline
chat's prompts and simulator so differences in traces come from instrumentation,
not a different conversation. No local reporting callback is attached.

AI attribution: Generated with AI assistance.
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.langfuse_runtime import launch
from lg_report.model_config import configured_model
from samples.simple_chat.app import SYSTEM_PROMPT, USER_PROMPTS
from samples.simple_chat.simulation import make_simulated_model


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Build the chat graph around a supplied live or simulated model.

    Construction makes no model calls. The in-memory backend prevents built-in
    file tools from accessing the developer's workspace. DeepAgents still sends
    its tool definitions even though the prompt asks for direct answers.
    """
    return create_deep_agent(
        model=model,
        backend=StateBackend(),
        subagents=[],
        name="chat-agent",
        system_prompt=SYSTEM_PROMPT,
    )


def create_graph(live: bool) -> CompiledStateGraph:
    """Select the explicitly requested model mode and compile this sample's graph."""
    return build_agent(configured_model()[0] if live else make_simulated_model())


def main() -> None:
    """Launch this sample with its own configuration and shared Langfuse lifecycle."""
    launch(
        app_file=__file__,
        description=__doc__,
        create_graph=create_graph,
        prompts=USER_PROMPTS,
        trace_name="simple-chat-langfuse",
    )


if __name__ == "__main__":
    main()
