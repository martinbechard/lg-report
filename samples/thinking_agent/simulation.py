"""Reproducible investigation: collect evidence, reason, then verify.

The large reasoning count is intentionally synthetic so students can see its
cost in the chart. The brief thinking text is illustrative, not a transcript
of those tokens. It is not added to the next prompt; visible output is.
"""

from langchain_core.messages import AIMessage

from lg_report.runner import MeteredDemoModel


def make_simulated_model() -> MeteredDemoModel:
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
