"""Demonstrate concrete judge feedback improving a deliberately shallow draft.

The incident and measurements below are fictional supplied evidence, not claims
about a real system. Fixed verdicts verify graph routing; live judge quality must
be evaluated separately. Neither agent imports these questions or responses.
AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.messages import AIMessage

from lg_report.agents.evidence_judge import SYSTEM_PROMPT as JUDGE_INSTRUCTIONS
from lg_report.agents.review_author import SYSTEM_PROMPT as AUTHOR_INSTRUCTIONS
from lg_report.platform.shared_simulated_model import SharedSimulatedModel

# Evidence is supplied by the client as part of the request. E1-E4 give the judge
# a concrete basis for checking claims without needing a retrieval service. The
# request explicitly asks for operational detail, so a generic overview falls short.
USER_PROMPTS = [
    (
        "Recommend a concrete latency fix, explaining evidence, rollout, verification, "
        "freshness risk, and rollback. Use these fictional incident notes: [E1] p95 "
        "latency is 1.8 seconds at 240 requests/second; repeated account lookups are "
        "observed. [E2] The database pool has 12 connections; no more database nodes "
        "are allowed. [E3] Account data may be cached for at most 60 seconds; changes "
        "must invalidate it. [E4] Target p95 is under 500 ms. There are no benchmark "
        "results yet. Distinguish a proposed improvement from measured success."
    )
]
# Deliberately incomplete, not deliberately false: this demonstrates that a
# reasonable-sounding answer can still fail the user's requested level of detail.
FIRST_DRAFT = (
    "Reduce redundant database access and consider caching account data. Roll "
    "out cautiously, monitor latency and freshness, and keep a rollback path."
)
# Feedback names repairable gaps instead of merely assigning a low score. Each
# item has something the next author response can address and the judge can check.
# The graph receives this as model output; it does not hardcode these corrections.
FIRST_REVIEW = {
    "verdict": "revise",
    "rationale": "The overview is plausible but lacks the requested operational detail and evidence links.",
    "feedback": [
        "Tie the diagnosis to E1/E2 and explain how the change avoids extra database nodes.",
        "Specify a proposed cache TTL within E3 and invalidate entries when accounts change.",
        "Give a staged rollout and a load-test acceptance criterion using E1/E4; do not claim unmeasured success.",
        "Specify a rollback trigger and the mechanism for restoring the previous path.",
    ],
}
# This candidate addresses the four concerns above. Specific rollout percentages
# and a 30-second TTL are labelled proposals, not facts supposedly measured in E1-E4.
# That distinction is important: adding invented evidence would not be improvement.
REVISED_DRAFT = (
    "Diagnosis: repeated account lookups coincide with p95 1.8s at 240 requests/s "
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
    "staging before canary deployment. No database nodes or schema changes are proposed."
)
# Approval closes the loop only because the validated verdict says approve. The
# graph does not know that this is the fixture's last response or expect two rounds.
FINAL_REVIEW = {
    "verdict": "approve",
    "rationale": "The revision links E1-E4 to a concrete proposal, respects the cache limit, and supplies measurable checks and rollback without inventing results.",
    "feedback": [],
}


def make_simulated_model():
    """Return a fresh shared simulator for the complete two-round scenario.

    The simulator recognizes each tool-free role by its actual system instruction.
    The author consumes its two answers in order, and the judge consumes its two
    verdicts. Their histories/token counters stay separate despite one LLM object.
    This exercises real StateGraph transitions but does not evaluate whether a
    provider would generate this feedback or approve this revision on its own.

    To test continued rejection, add author/review responses and change the last
    verdict to revise. To test immediate approval, supply one answer and approval.
    The simulator raises on exhausted scripts rather than recycling old answers.
    """
    # AIMessage is the same result shape used by provider adapters. JSON encoding
    # ensures the judge goes through the real parse/validation boundary rather
    # than passing a trusted Python Review object directly into workflow state.
    return SharedSimulatedModel(
        scripts={
            AUTHOR_INSTRUCTIONS: [
                AIMessage(content=FIRST_DRAFT),
                AIMessage(content=REVISED_DRAFT),
            ],
            JUDGE_INSTRUCTIONS: [
                AIMessage(content=json.dumps(FIRST_REVIEW)),
                AIMessage(content=json.dumps(FINAL_REVIEW)),
            ],
        },
        metadata={
            "report_effort": "light",
            "report_description": "Perform the current author or evidence-review role.",
        },
    )
