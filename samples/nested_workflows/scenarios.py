"""Author deterministic agent decisions, without pretending they are live tests.

The default story deliberately includes a reviewer missing an acceptance defect;
that is how the later tester sends work back into the coding context. Code and
assessments are fixture data. The workflow, not this module, executes transitions.
This module has no LangGraph dependency so the fixture contracts can be checked
independently of the integration environment.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from copy import deepcopy

from agent_runtime.workflows.nested_policy import Limits

TASK = {
    "objective": "Implement normalize_tags(tags) for a list of strings.",
    "acceptance": [
        "Trim whitespace and convert tags to lowercase.",
        "Remove duplicates while preserving first-occurrence order.",
        "Discard empty and whitespace-only tags.",
        "Do not modify the caller's input list.",
    ],
    "constraints": ["Use only the Python standard library."],
}
USER_PROMPTS = [
    (
        "Implement normalize_tags(tags). Trim and lowercase tags, deduplicate while "
        "preserving order, discard blank tags, and do not mutate the input. Use the "
        "Python standard library only. OUTER_ONLY_DETAIL: an incidental planning "
        "briefing marker, not part of the implementation assignment."
    )
]
SOURCE_V1 = """def normalize_tags(tags):
    return [tag.strip().lower() for tag in tags]
"""
SOURCE_V2 = """def normalize_tags(tags):
    return list(dict.fromkeys(tag.strip().lower() for tag in tags))
"""
SOURCE_V3 = """def normalize_tags(tags):
    cleaned = (tag.strip().lower() for tag in tags)
    return list(dict.fromkeys(tag for tag in cleaned if tag))
"""


def script(options=None):
    """Return fresh queues for rework, either breaker path, or immediate success."""
    options = options or {}
    scenario = options.get("scenario", "rework")
    limits = Limits(
        options.get("max_review_rounds", 3), options.get("max_coding_cycles", 3)
    )
    queues = {
        name: []
        for name in ("planner", "coding_supervisor", "coder", "reviewer", "tester")
    }
    current = 0

    def plan(action, note):
        queues["planner"].append(
            {"action": action, "task": deepcopy(TASK), "note": note}
        )

    def supervise(action, directive):
        queues["coding_supervisor"].append(
            {
                "action": action,
                "directive": "CODING_ONLY_DETAIL: " + directive,
            }
        )

    def develop(source):
        nonlocal current
        current += 1
        queues["coder"].append(
            {
                "candidate_id": f"candidate-{current}",
                "source": source,
                "working_notes": "CODING_ONLY_DETAIL: private implementation discussion.",
            }
        )

    def assess(role, verdict, findings, evidence):
        queues[role].append(
            {
                "candidate_id": f"candidate-{current}",
                "verdict": verdict,
                "findings": findings,
                "evidence": evidence,
                "evidence_kind": "scripted",
            }
        )

    def accepted_review():
        assess(
            "reviewer",
            "pass",
            [],
            "Scripted review approval; not proof of correctness.",
        )
        supervise("return", "Return the review-approved artifact to the planner.")
        plan(
            "test", "Send the candidate to the tester; review alone is not acceptance."
        )

    plan("code", "Dispatch the stable task to the coding supervisor.")
    if scenario == "rework":
        if limits.max_review_rounds < 2 or limits.max_coding_cycles < 2:
            raise ValueError(
                "The rework story requires at least two rounds and two cycles"
            )
        supervise("code", "Implement the task.")
        develop(SOURCE_V1)
        assess(
            "reviewer",
            "fail",
            ["Remove duplicate normalized tags."],
            "Scripted finding: duplicate tags remain in candidate-1.",
        )
        supervise("code", "Address the review finding about duplicate tags.")
        develop(SOURCE_V2)
        accepted_review()
        assess(
            "tester",
            "fail",
            [
                "For ['  ', 'Alpha', 'alpha'], return ['alpha'], not ['', 'alpha']; remove blank tags."
            ],
            "Scripted acceptance defect, not an executed test receipt.",
        )
        plan("code", "Return tester defects to the existing coding context.")
        supervise("code", "Continue the same task and remove blank tags.")
        develop(SOURCE_V3)
        accepted_review()
        assess(
            "tester",
            "pass",
            [],
            "Scripted acceptance approval; no test runner was called.",
        )
        plan("finish", "The repaired candidate passed the scripted delivery gates.")
    elif scenario == "review_limit":
        for _ in range(limits.max_review_rounds):
            supervise("code", "Correct duplicate handling before returning.")
            develop(SOURCE_V1)
            assess(
                "reviewer",
                "fail",
                ["Remove duplicate normalized tags."],
                "The scripted reviewer continues to find duplicates.",
            )
        supervise("escalate", "Stop: the maximum review rounds have been used.")
        plan(
            "escalate", "The coding supervisor returned a blocked result; do not test."
        )
    elif scenario == "test_limit":
        for cycle in range(limits.max_coding_cycles):
            supervise("code", "Address the task and any returned tester defects.")
            develop(SOURCE_V2)
            accepted_review()
            assess(
                "tester",
                "fail",
                ["Whitespace-only input still produces an empty tag."],
                "The scripted acceptance defect persists; no test runner was called.",
            )
            if cycle + 1 < limits.max_coding_cycles:
                plan("code", "Re-enter coding with the tester's defect packet.")
            else:
                plan(
                    "escalate",
                    "Stop: the maximum test-fix coding cycles have been used.",
                )
    elif scenario == "first_pass":
        supervise("code", "Implement all acceptance criteria.")
        develop(SOURCE_V3)
        accepted_review()
        assess(
            "tester",
            "pass",
            [],
            "Scripted acceptance approval; no test runner was called.",
        )
        plan("finish", "The initial candidate passed both scripted gates.")
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return queues
