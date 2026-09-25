"""Author the thinking agent example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. The shared catalog
constructs the offline model from CONVERSATION; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "thinking_agent",
    "name": "Thinking agent",
    "description": "Investigate evidence and test a hypothesis.",
}

# Reasoning counts and thinking_text below are illustrative fixture metadata,
# not private reasoning captured from a provider or a real benchmark.
CONVERSATION = [
    {
        "role": "client",
        "content": "Investigate service latency and validate a plan meeting the constraints.",
    },
    {
        "role": "investigation_agent",
        "content": "Inspect the traffic evidence.",
        "tool_calls": [
            {
                "name": "inspect_service",
                "args": {"section": "traffic"},
                "id": "inspect-0",
            }
        ],
    },
    {
        "role": "tool",
        "name": "inspect_service",
        "tool_call_id": "inspect-0",
        "content": "Expected: actual inspect_service result for {'section': 'traffic'}; supplied by the running "
        "tool.",
    },
    {
        "role": "investigation_agent",
        "content": "Inspect the database evidence.",
        "tool_calls": [
            {
                "name": "inspect_service",
                "args": {"section": "database"},
                "id": "inspect-1",
            }
        ],
    },
    {
        "role": "tool",
        "name": "inspect_service",
        "tool_call_id": "inspect-1",
        "content": "Expected: actual inspect_service result for {'section': 'database'}; supplied by the running "
        "tool.",
    },
    {
        "role": "investigation_agent",
        "content": "Inspect the constraints evidence.",
        "tool_calls": [
            {
                "name": "inspect_service",
                "args": {"section": "constraints"},
                "id": "inspect-2",
            }
        ],
    },
    {
        "role": "tool",
        "name": "inspect_service",
        "tool_call_id": "inspect-2",
        "content": "Expected: actual inspect_service result for {'section': 'constraints'}; supplied by the "
        "running tool.",
    },
    {
        "role": "investigation_agent",
        "content": "Proposed plan: deduplicate account lookups and add a 30-second invalidated cache. Validate "
        "load, freshness, and rollback.",
        "tool_calls": [
            {
                "name": "test_plan",
                "args": {"check": "load"},
                "id": "test-0",
            }
        ],
        "response_metadata": {
            "simulated_reasoning_tokens": 12000,
            "thinking_text": "Compare the evidence against the latency and freshness "
            "constraints. Evaluate deduplication and short-lived caching, then "
            "verify load, invalidation, and rollback.",
        },
    },
    {
        "role": "tool",
        "name": "test_plan",
        "tool_call_id": "test-0",
        "content": "Expected: actual test_plan result for {'check': 'load'}; supplied by the running tool.",
    },
    {
        "role": "investigation_agent",
        "content": "Verify freshness before recommending the change.",
        "tool_calls": [
            {
                "name": "test_plan",
                "args": {"check": "freshness"},
                "id": "test-1",
            }
        ],
        "response_metadata": {
            "simulated_reasoning_tokens": 600,
            "thinking_text": "Review the freshness requirement and select a verification check "
            "for the proposed change.",
        },
    },
    {
        "role": "tool",
        "name": "test_plan",
        "tool_call_id": "test-1",
        "content": "Expected: actual test_plan result for {'check': 'freshness'}; supplied by the running tool.",
    },
    {
        "role": "investigation_agent",
        "content": "Verify rollback before recommending the change.",
        "tool_calls": [
            {
                "name": "test_plan",
                "args": {"check": "rollback"},
                "id": "test-2",
            }
        ],
        "response_metadata": {
            "simulated_reasoning_tokens": 600,
            "thinking_text": "Review the rollback requirement and select a verification check "
            "for the proposed change.",
        },
    },
    {
        "role": "tool",
        "name": "test_plan",
        "tool_call_id": "test-2",
        "content": "Expected: actual test_plan result for {'check': 'rollback'}; supplied by the running tool.",
    },
    {
        "role": "investigation_agent",
        "content": "Recommend deduplicated lookups and a 30-second invalidated cache. The load test meets the "
        "500ms target at 410ms; freshness and rollback checks pass. Roll out behind the feature flag "
        "and monitor latency and pool occupancy.",
        "response_metadata": {
            "simulated_reasoning_tokens": 1800,
            "thinking_text": "Review the verification results against the acceptance criteria "
            "and summarize the recommendation with its rollout controls.",
        },
    },
]
