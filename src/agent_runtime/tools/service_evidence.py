"""Fictional service evidence and deterministic verification tools.

These functions are teaching fixtures, not production diagnostics or load tests.
Keeping them separate makes the graph contract and the evidence boundary clear.

AI attribution: Generated with AI assistance.

Design: docs/chat-composition.md.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.tools import tool

# These tools intentionally expose a finite fixture contract. Keeping the
# dictionaries inside each function prevents shared mutation and makes unknown
# keys fail loudly rather than returning guessed evidence.


# The docstrings below are tool descriptions sent to the model, so keep
# implementation guidance here rather than expanding the model-facing prompt.
# Valid sections are traffic, database, and constraints. Unknown sections raise
# KeyError instead of silently returning unrelated evidence. No host is inspected.
@tool(parse_docstring=True)
def inspect_service(section: str) -> str:
    """Read fictional service evidence; no live service is inspected.

    Args:
        section: Evidence section to retrieve: traffic, database, or constraints.
            Use one of these exact lowercase names.
    """
    # The investigator calls this to obtain the selected evidence category
    # before proposing a plan. Its returned string is the tool observation the
    # next model request receives, rather than the investigator's final answer.
    # No network client, clock, or service handle is consulted here. The fixed
    # values teach the tool-call boundary while keeping reports reproducible.
    evidence_by_section = {
        "traffic": "Peak 240 requests/s; p95 1.8s; 12 workers. Latency rises during database bursts.",
        "database": "Pool limit 12. Median query 80ms; p95 900ms. Duplicate account lookups appear per request.",
        "constraints": "Target p95 under 500ms. No additional database nodes. Cache lifetime must not exceed 60s.",
    }
    return evidence_by_section[section]


# Valid checks are load, freshness, and rollback. These fixed results keep the
# lesson reproducible; a PASS describes the fixture, not a real benchmark.
# Unknown checks raise KeyError. No workload or rollback is executed on this PC.
@tool(parse_docstring=True)
def test_plan(check: str) -> str:
    """Return a fixed fictional plan-check result; no real test is executed.

    Args:
        check: Check result to retrieve: load, freshness, or rollback.
            Use one of these exact lowercase names.
    """
    # The investigator calls this after considering a plan so it can discuss
    # the lesson's verification outcomes. `check` selects a scenario result;
    # no candidate plan is accepted or evaluated by this function.
    # A PASS records the sample fixture's expected result; it is not a claim
    # that this machine ran a benchmark or changed a rollback flag.
    results_by_check = {
        "load": "PASS: deduplicated lookups plus 30s cache yield p95 410ms at 240 requests/s; pool occupancy 65%.",
        "freshness": "PASS: account changes invalidate cached values; maximum fallback age is 30s.",
        "rollback": "PASS: feature flag restores original lookup path; no schema migration required.",
    }
    return results_by_check[check]
