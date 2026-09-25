"""Show a parent assignment followed by a child drafting and review conversation.

CONVERSATION is the single authored source for the client prompt and simulated
role replies. Live mode uses the same graphs with provider responses instead.
The code below is fixture text: neither mode executes the proposed function.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

SAMPLE = {
    "id": "nested_workflows",
    "name": "Nested workflows",
    "description": "A parent dispatches an assignment to a child author/judge review loop.",
    "options": {"max_rounds": 3},
}

CONVERSATION = [
    {
        "role": "client",
        "content": "Implement normalize_tags(tags). Trim and lowercase strings, remove "
        "duplicates while preserving order, discard blank tags, and do not mutate "
        "the input. Use the Python standard library only.",
    },
    {
        "role": "work_planner",
        "content": "Write normalize_tags(tags) using only the Python standard library. "
        "Return a new list of trimmed, lowercase, nonblank strings, deduplicated in "
        "first-occurrence order. Preserve the caller's input list.",
    },
    {
        "role": "review_author",
        "content": "def normalize_tags(tags):\n"
        "    return [tag.strip().lower() for tag in tags]\n",
    },
    {
        "role": "evidence_judge",
        "content": json.dumps(
            {
                "verdict": "revise",
                "rationale": "The draft trims and lowercases but retains duplicates and blanks.",
                "feedback": [
                    "Remove duplicate normalized tags while preserving order.",
                    "Discard empty normalized tags.",
                ],
            }
        ),
    },
    {
        "role": "review_author",
        "content": "def normalize_tags(tags):\n"
        "    cleaned = (tag.strip().lower() for tag in tags)\n"
        "    return list(dict.fromkeys(tag for tag in cleaned if tag))\n",
    },
    {
        "role": "evidence_judge",
        "content": json.dumps(
            {
                "verdict": "approve",
                "rationale": "The revised source meets the assignment by inspection; no tests were executed.",
                "feedback": [],
            }
        ),
    },
]
