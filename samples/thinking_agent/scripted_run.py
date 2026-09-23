"""Reproducible investigation: collect evidence, reason, then verify.

The large reasoning count is intentionally synthetic so students can see its
cost in the chart. The brief thinking text is illustrative, not a transcript
of those tokens. It is not added to the next prompt; visible output is.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# LangChain's AIMessage holds an assistant response; constructing it runs nothing.
# Its tool_calls, when present, are proposed names/arguments, not tool results.
from langchain_core.messages import AIMessage

from agent_runtime.harness.simulated_model import MeteredDemoModel


def make_simulated_model() -> MeteredDemoModel:
    """Make the cost of a large reasoning step visible in a repeatable investigation.

    Return a fresh seven-response fixture with a deliberate reasoning peak.

    Three evidence calls precede three verification calls and the final answer.
    This predictable sequence lets a learner correlate report request numbers
    with graph activity. Reasoning counts are scenario inputs; visible response
    and growing input counts are measured by MeteredDemoModel. No API is called.
    """
    # These visible sentences announce actions; they are ordinary assistant
    # output, not thinking text. LangGraph executes each requested tool before
    # asking this model for the next scripted response.
    responses = []
    for index, section in enumerate(["traffic", "database", "constraints"]):
        responses.append(
            AIMessage(
                content=f"Inspect the {section} evidence.",
                tool_calls=[
                    {
                        "name": "inspect_service",
                        "args": {"section": section},
                        "id": f"inspect-{index}",
                    }
                ],
            )
        )
    # Put the largest reasoning charge at the transition from evidence to plan.
    # The following checks produce additional model calls, showing that one
    # user turn can have a costly decision followed by several cheaper decisions.
    # All three index == 0 choices below identify the first verification request,
    # after evidence collection. It introduces the plan, carries the large
    # reasoning charge, and describes the broad decision. Later requests reuse
    # that plan and perform one cheaper, narrower check each.
    for index, check in enumerate(["load", "freshness", "rollback"]):
        responses.append(
            AIMessage(
                content=(
                    "Proposed plan: deduplicate account lookups and add a 30-second invalidated cache. Validate load, freshness, and rollback."
                    if index == 0
                    else f"Verify {check} before recommending the change."
                ),
                tool_calls=[
                    {
                        "name": "test_plan",
                        "args": {"check": check},
                        "id": f"test-{index}",
                    }
                ],
                response_metadata={
                    # Display text and billed reasoning count are intentionally
                    # separate. This short fixture cannot reconstruct 12,000
                    # tokens of private reasoning from an actual provider.
                    "simulated_reasoning_tokens": 12000 if index == 0 else 600,
                    "thinking_text": (
                        "Compare the evidence against the latency and freshness constraints. Evaluate deduplication and short-lived caching, then verify load, invalidation, and rollback."
                        if index == 0
                        else f"Review the {check} requirement and select a verification check for the proposed change."
                    ),
                },
            )
        )
    responses.append(
        AIMessage(
            content="Recommend deduplicated lookups and a 30-second invalidated cache. The load test meets the 500ms target at 410ms; freshness and rollback checks pass. Roll out behind the feature flag and monitor latency and pool occupancy.",
            response_metadata={
                "simulated_reasoning_tokens": 1800,
                "thinking_text": "Review the verification results against the acceptance criteria and summarize the recommendation with its rollout controls.",
            },
        )
    )
    # Reasoning effort is a display annotation here, not a real provider setting.
    return MeteredDemoModel(
        responses=responses,
        metadata={
            "report_effort": "high",
            "report_description": "Investigate service evidence, evaluate a constrained improvement, and verify the proposed plan.",
        },
    )


# User input belongs to the test scenario, never the agent definition.
USER_PROMPTS = [
    "Investigate service latency and validate a plan meeting the constraints."
]


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}
