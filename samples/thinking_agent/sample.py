"""Author the thinking agent example in chronological conversation order.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

User messages and AI replies share one script, including multiple model/tool
steps within a user turn. Tool entries document expected observations only;
the workflow executes the actual tools. Model choices remain simulated.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.model_factory import client_prompts, model_responses
from agent_runtime.harness.simulated_model import MeteredDemoModel

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
        "role": "ai",
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
        "role": "ai",
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
        "role": "ai",
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
        "role": "ai",
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
        "role": "ai",
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
        "role": "ai",
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
        "role": "ai",
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

# User inputs and model replies are projections of the same ordered script.
USER_PROMPTS = client_prompts(CONVERSATION)


def make_simulated_model():
    """Extract AI replies into a fresh model with independent cursor and usage."""
    return MeteredDemoModel(
        responses=model_responses(CONVERSATION),
        metadata={
            "report_effort": "high",
            "report_description": "Investigate service evidence, evaluate a constrained "
            "improvement, and verify the proposed plan.",
            "lc_versions": {"langchain-core": "1.6.3", "langchain": "1.4.1"},
        },
    )


def build_scripted_models(options):
    """Let the catalog retain this sample's report metadata in simulated mode.

    The catalog explicitly calls this callback when building a simulated run.
    options merges sample defaults with run overrides; this fixed script ignores
    them. The workflow's build_model(caller="workflow") receives the "workflow"
    entry. Real mode bypasses this factory and uses the configured provider.
    """
    return {"workflow": make_simulated_model()}
