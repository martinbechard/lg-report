"""Define an agent that investigates latency using fictional service evidence.

Own instructions and tool registration; clients supply user requests at runtime.
The injected model can be a provider or a deterministic test fixture.

Design: docs/chat-composition.md.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from lg_report.tools.service_evidence import inspect_service, test_plan

# Role instructions apply to arbitrary user questions, not a fixed test case.
SYSTEM_PROMPT = "Use the fictional service evidence to propose and validate a latency improvement. Inspect traffic, database behavior, and constraints before testing the plan."


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Return an investigation graph around the supplied model without running it.

    Both tools remain fictional fixtures even when model is a live provider.
    This lets learners examine a longer reasoning/tool trace without connecting
    the agent to real infrastructure. StateBackend confines built-in file
    operations to graph state; callers attach recording when invoking the graph.
    """
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
        name="investigation_agent",
    )
