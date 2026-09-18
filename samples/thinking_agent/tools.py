"""Fictional service evidence and deterministic verification tools.

These functions are teaching fixtures, not production diagnostics or load tests.
Keeping them separate makes the graph contract and the evidence boundary clear.
"""

from langchain_core.tools import tool


@tool
def inspect_service(section: str) -> str:
    """Read a section of the fictional service's operational evidence."""
    evidence = {
        "traffic": "Peak 240 requests/s; p95 1.8s; 12 workers. Latency rises during database bursts.",
        "database": "Pool limit 12. Median query 80ms; p95 900ms. Duplicate account lookups appear per request.",
        "constraints": "Target p95 under 500ms. No additional database nodes. Cache lifetime must not exceed 60s.",
    }
    return evidence[section]


@tool
def test_plan(check: str) -> str:
    """Return a deterministic test result for the proposed service change."""
    results = {
        "load": "PASS: deduplicated lookups plus 30s cache yield p95 410ms at 240 requests/s; pool occupancy 65%.",
        "freshness": "PASS: account changes invalidate cached values; maximum fallback age is 30s.",
        "rollback": "PASS: feature flag restores original lookup path; no schema migration required.",
    }
    return results[check]
