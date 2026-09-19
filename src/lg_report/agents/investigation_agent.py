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

# The prompt describes a fixture-backed investigation. Keeping that boundary
# explicit prevents live-provider output from being mistaken for telemetry.


def build_agent(model: BaseChatModel) -> CompiledStateGraph:
    """Prepare a latency investigator for a reproducible evidence-and-plan lesson.

    Both tools remain fictional fixtures even when model is a live provider.
    This lets learners examine a longer reasoning/tool trace without connecting
    the agent to real infrastructure. StateBackend confines built-in file
    operations to graph state; callers attach recording when invoking the graph.
    Tool or model errors propagate so an incomplete investigation cannot
    silently become a successful recommendation.
    """
    # A proposed model tool call becomes an actual local function invocation
    # through the graph's tool loop. Its string result becomes a ToolMessage
    # for the next model request. The overall graph returns message state after
    # the model finishes, not a single test result or a live service measurement.
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
