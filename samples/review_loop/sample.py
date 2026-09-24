"""Demonstrate concrete judge feedback improving a deliberately shallow draft.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

The incident and measurements below are fictional supplied evidence, not claims
about a real system. Fixed verdicts verify graph routing; live judge quality must
be evaluated separately. Neither agent imports these questions or responses.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from agent_runtime.harness.simulated_model import SimulatedModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "review_loop",
    "name": "Review loop",
    "description": "An author revises a draft using a judge's feedback.",
    "options": {"max_rounds": 3, "first_draft_high_level": True},
}


# Both roles read their replies from this one chronological exchange. The judge
# emits JSON so the real workflow still parses and validates every verdict.
CONVERSATION = [
    {
        "role": "client",
        "content": "Recommend a concrete latency fix, explaining evidence, rollout, verification, freshness risk, and rollback. Use these fictional incident notes: [E1] p95 latency is 1.8 seconds at 240 requests/second; repeated account lookups are observed. [E2] The database pool has 12 connections; no more database nodes are allowed. [E3] Account data may be cached for at most 60 seconds; changes must invalidate it. [E4] Target p95 is under 500 ms. There are no benchmark results yet. Distinguish a proposed improvement from measured success.",
    },
    # This deliberately shallow draft gives the judge concrete gaps to identify.
    {
        "role": "review_author",
        "content": "Reduce redundant database access and consider caching account data. Roll "
        "out cautiously, monitor latency and freshness, and keep a rollback path.",
    },
    {
        "role": "evidence_judge",
        "content": json.dumps(
            {
                "verdict": "revise",
                "rationale": "The overview is plausible but lacks the requested operational detail and evidence links.",
                "feedback": [
                    "Tie the diagnosis to E1/E2 and explain how the change avoids extra database nodes.",
                    "Specify a proposed cache TTL within E3 and invalidate entries when accounts change.",
                    "Give a staged rollout and a load-test acceptance criterion using E1/E4; do not claim unmeasured success.",
                    "Specify a rollback trigger and the mechanism for restoring the previous path.",
                ],
            }
        ),
    },
    {
        "role": "review_author",
        "content": "Diagnosis: repeated account lookups coincide with p95 1.8s at 240 requests/s "
        "[E1]; the pool is limited to 12 connections and more nodes are disallowed [E2]. "
        "This supports testing reduced lookup demand, but does not prove causation.\n\n"
        "Proposed fix: deduplicate account lookups within each request, then test a "
        "30-second account cache. Invalidate affected entries on account changes; this "
        "TTL stays below the 60-second ceiling [E3]. Cache invalidation failures remain "
        "a freshness risk, so test update/read races before rollout.\n\n"
        "Proposed rollout: use a feature flag, canary at 5% traffic, then 25% and 100% "
        "only after each stage passes checks. At 240 requests/s [E1], require p95 below "
        "500ms [E4], no observed stale reads after acknowledged account updates, and "
        "no increase in errors relative to baseline. Monitor pool saturation as well. "
        "These are proposed checks, not benchmark results; none exist yet.\n\n"
        "Rollback: disable the cache/deduplication flag to restore the original lookup "
        "path if stale reads occur or latency/errors regress. Validate this switch in "
        "staging before canary deployment. No database nodes or schema changes are proposed.",
    },
    {
        "role": "evidence_judge",
        "content": json.dumps(
            {
                "verdict": "approve",
                "rationale": "The revision links E1-E4 to a concrete proposal, respects the cache limit, and supplies measurable checks and rollback without inventing results.",
                "feedback": [],
            }
        ),
    },
]


def make_simulated_model():
    """Extract each role's replies while retaining separate shared-model ledgers."""
    return SimulatedModel(
        conversation=CONVERSATION,
        metadata={
            "report_effort": "light",
            "report_description": "Perform the current author or evidence-review role.",
        },
    )


def build_scripted_models(options):
    """Retain role routing in simulated mode; this fixed story ignores options."""
    return {"workflow": make_simulated_model()}
