"""Script file-tool decisions for the context-budget planning exercise.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

Real native tools write and read the plan, code, and tests. Scripted model
responses make the lesson deterministic; compaction remains native middleware
behavior, and reported usage is illustrative rather than provider billing.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from agent_runtime.harness.simulated_model import SimulatedModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "context_budget",
    "name": "Context budgets",
    "description": "File-backed planner and worker with shared compaction and an isolated "
    "code reviewer.",
}


# Read down this list to follow dispatch, implementation, independent review,
# and plan completion. The second client entry starts the boundary-test request.
# Tool entries describe expectations only; source changes and reviews run in the
# real graph. The summary entry is an on-demand response, not a scheduled turn.
# Repetition deliberately fills context to exercise native compaction. Keeping
# the multiplier visible explains that load without hundreds of duplicate lines.
CONVERSATION = [
    {
        "role": "client",
        "content": (
            "Build a small, well-tested Python slug function. Make a detailed file plan first. "
            + "The implementation should be inspectable and independent of services. "
            * 100
        ),
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/plan.md",
                    "content": """# Slug utility delivery plan

Goal: provide one deterministic Python function named slugify in /slug.py. It turns a title into a lowercase ASCII slug suitable for a local document name. The sample must demonstrate file-backed coordination and independent review, not a production packaging workflow. /test_slug.py contains focused unittest coverage. Each step follows worker implementation, independent approval, then planner completion before the next step starts.

Working agreement: every task has a stable ID. An edit changes only the named task status and evidence. Pending means no artifact is claimed. Complete means the current version was independently approved and the planner recorded that approval. Blocked means a concrete missing input or failed operation. A reviewer report is evidence of inspection, not evidence that tests executed. The reviewer has read-only access and a separate context; its assignment must include all paths and intended behavior. The planner retains ownership of the plan. The worker reports implementation evidence but does not edit the plan.

Design contract: accept a string title. Normalize accented Latin letters to their ASCII base where Unicode decomposition provides one. Convert to lowercase. Treat runs of punctuation and whitespace as a single hyphen. Strip leading and trailing hyphens. Empty or separator-only text returns an empty string. A non-string input raises TypeError with a clear message. Do not use network calls, environment configuration, or generated identifiers. Preserve deterministic output for the same input.

Implementation notes: prefer unicodedata.normalize over a third-party transliteration package. Encode decomposed text as ASCII while dropping characters that have no ASCII form. Explain that limitation in a docstring because some scripts become empty. Use a regular expression for separator runs and make its pattern explicit. Keep the public function side-effect free. Avoid broad exception handlers and hidden fallback behavior. Include a purpose header, AI attribution, and the project copyright in each Python file.

Verification agreement: tests use only Python's standard library unittest. They should verify a normal title, repeated whitespace and punctuation, accented Latin input, leading and trailing separators, empty input, and invalid type. Tests are source artifacts until a command actually executes them; the agent in this workflow has file tools but no shell tool. The final report must distinguish written tests from passing tests. Review findings should name a triggering example and the affected task.

Task T1 — implement the public function. Status: pending. Acceptance: /slug.py exposes slugify; ordinary words become lowercase with one hyphen; accents with Latin decompositions lose marks; punctuation runs do not create repeated hyphens; non-string values fail visibly. Read the plan after recording completion.

Task T2 — create focused tests. Status: pending. Acceptance: /test_slug.py imports slugify and uses unittest with examples for normal titles, spacing, punctuation, accents, empty input, and invalid input. The tests should distinguish a plausible broken implementation from the expected contract. Read the plan after recording completion.

Risk notes: file reads may be paginated, so inspect the rest if a native read is truncated. A successful write does not prove behavior; it only proves content was saved. A summary may omit exact plan lines, so the current file controls status after compaction. If a tool reports an error, keep the task pending or blocked and explain why. Avoid broadening the task to packaging, CLI behavior, or external deployment. The intended learning is the relationship among plan state, conversation state, reviewer isolation, and context limits.
""",
                },
                "id": "write_file-1",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        "tool_call_id": "write_file-1",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": json.dumps(
            {
                "action": "work",
                "task_id": "T1",
                "files": ["/slug.py"],
                "message": "Implement T1 only: write /slug.py according to /plan.md. Tests belong to "
                "the next step.",
            }
        ),
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-20",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-20",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/slug.py",
                    "content": '''"""Create deterministic ASCII slugs for local document titles.

Accents with Latin decomposition are removed; characters without ASCII
representation are dropped. The function has no filesystem or network effects.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import re
import unicodedata


def slugify(title: str) -> str:
    """Return a lowercase, hyphen-separated ASCII form of a title."""
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    # Decompose Latin accents before the ASCII boundary; unsupported scripts
    # are dropped rather than transliterated through an undeclared dependency.
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower())
''',
                },
                "id": "write_file-21",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        "tool_call_id": "write_file-21",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "T1 candidate written; source review required. Tests not executed.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-40",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-40",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/slug.py"},
                "id": "read_file-41",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-41",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": json.dumps(
            {
                "task_id": "T1",
                "verdict": "revise",
                "evidence": 'slugify("! Hi !") returns "-hi-". Strip boundary hyphens in /slug.py. '
                "Tests not executed.",
            }
        ),
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/slug.py"},
                "id": "read_file-22",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-22",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/slug.py",
                    "old_string": "ascii_title.lower())",
                    "new_string": 'ascii_title.lower()).strip("-")',
                },
                "id": "edit_file-23",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_file",
        "tool_call_id": "edit_file-23",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "T1 repaired: boundary hyphens are stripped. Please review the current file.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-43",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-43",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/slug.py"},
                "id": "read_file-44",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-44",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": json.dumps(
            {
                "task_id": "T1",
                "verdict": "approve",
                "evidence": "/slug.py now strips boundary hyphens and satisfies T1. Source inspection "
                "only; tests not executed.",
            }
        ),
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-2",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-2",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/plan.md",
                    "old_string": "Task T1 — implement the public function. Status: pending.",
                    "new_string": "Task T1 — implement the public function. Status: complete. Review "
                    "evidence: independent reviewer approved current source; tests not "
                    "executed.",
                },
                "id": "edit_file-3",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_file",
        "tool_call_id": "edit_file-3",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-4",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-4",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": json.dumps(
            {
                "action": "work",
                "task_id": "T2",
                "files": ["/test_slug.py"],
                "message": "Implement T2 only: write /test_slug.py with focused unittest coverage of "
                "the plan contract. Read current /slug.py first.",
            }
        ),
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-24",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-24",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/slug.py"},
                "id": "read_file-25",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-25",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "write_file",
                "args": {
                    "file_path": "/test_slug.py",
                    "content": '''"""Check the public slug contract with standard-library unittest.

These are source tests; the sample's file-only agents do not execute them.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import unittest

from slug import slugify


class SlugTests(unittest.TestCase):
    """Exercise distinct behaviors named in the plan."""

    def test_normal_title(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_runs_of_separators(self):
        self.assertEqual(slugify("Many   spaces... and punctuation!"), "many-spaces-and-punctuation")

    def test_latin_accent(self):
        self.assertEqual(slugify("Café Étude"), "cafe-etude")

    def test_empty(self):
        self.assertEqual(slugify(""), "")

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            slugify(None)
''',
                },
                "id": "write_file-26",
            }
        ],
    },
    {
        "role": "tool",
        "name": "write_file",
        "tool_call_id": "write_file-26",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "T2 test source written; tests not executed. Review required.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-46",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-46",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/slug.py"},
                "id": "read_file-47",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-47",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/test_slug.py"},
                "id": "read_file-48",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-48",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": json.dumps(
            {
                "task_id": "T2",
                "verdict": "approve",
                "evidence": "/test_slug.py covers normal input, punctuation, accents, empty input and "
                "invalid types as T2 requires. Tests not executed.",
            }
        ),
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-5",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-5",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/plan.md",
                    "old_string": "Task T2 — create focused tests. Status: pending.",
                    "new_string": "Task T2 — create focused tests. Status: complete. Review evidence: "
                    "independent reviewer approved current source; tests not executed.",
                },
                "id": "edit_file-6",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_file",
        "tool_call_id": "edit_file-6",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-7",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-7",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": json.dumps(
            {
                "action": "finish",
                "task_id": "",
                "files": [],
                "message": "T1 and T2 independently approved and complete. Tests not executed.",
            }
        ),
    },
    {
        "role": "client",
        "content": "Continue the plan: strengthen the boundary tests, get an independent review, and record the "
        "outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-8",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-8",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/plan.md",
                    "old_string": "Risk notes:",
                    "new_string": "Task T3 — extend boundary coverage. Status: pending. Acceptance: add separator-only and leading/trailing punctuation tests to /test_slug.py."
                    + "\n\nRisk notes:",
                },
                "id": "edit_file-9",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_file",
        "tool_call_id": "edit_file-9",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": json.dumps(
            {
                "action": "work",
                "task_id": "T3",
                "files": ["/test_slug.py"],
                "message": "Implement T3 only: extend /test_slug.py with separator-only and "
                "outer-punctuation tests. Preserve existing tests.",
            }
        ),
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-27",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-27",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/test_slug.py"},
                "id": "read_file-28",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-28",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/test_slug.py",
                    "old_string": "    def test_invalid_input(self):",
                    "new_string": """
    def test_only_separators(self):
        self.assertEqual(slugify("...  ---"), "")

    def test_outer_punctuation(self):
        self.assertEqual(slugify("!! Great Day ??"), "great-day")
"""
                    + "\n    def test_invalid_input(self):",
                },
                "id": "edit_file-29",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_file",
        "tool_call_id": "edit_file-29",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "worker",
        "content": "T3 boundary cases added; tests not executed. Review required.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-49",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-49",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/slug.py"},
                "id": "read_file-50",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-50",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/test_slug.py"},
                "id": "read_file-51",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-51",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "isolated-reviewer",
        "content": json.dumps(
            {
                "task_id": "T3",
                "verdict": "approve",
                "evidence": "/test_slug.py adds test_only_separators and test_outer_punctuation; "
                "existing tests preserved. Tests not executed.",
            }
        ),
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-10",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-10",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/plan.md",
                    "old_string": "Task T3 — extend boundary coverage. Status: pending.",
                    "new_string": "Task T3 — extend boundary coverage. Status: complete. Review "
                    "evidence: independent reviewer approved current source; tests not "
                    "executed.",
                },
                "id": "edit_file-11",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_file",
        "tool_call_id": "edit_file-11",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/plan.md"},
                "id": "read_file-12",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "tool_call_id": "read_file-12",
        "content": "Expected: actual file-tool result; reads show current workspace content, writes/edits report "
        "their outcome.",
    },
    {
        "role": "planner",
        "content": json.dumps(
            {
                "action": "finish",
                "task_id": "",
                "files": [],
                "message": "All assigned steps were independently approved and recorded complete. Tests "
                "were written but not executed.",
            }
        ),
    },
    {
        "role": "workflow-summary",
        "content": "The plan lives in /plan.md. Planner owns only plan updates; worker owns source and tests; reviewer independently approves each current task. The workflow supplies the authoritative current assignment and verdict after this summary. Never infer approval or the next actor from this summary.",
        "when": "On demand when middleware compacts context; not a fixed step in the exchange.",
    },
]


def make_simulated_models():
    """Extract independent role queues without losing compaction-aware metering."""
    roles = {
        "planner": "Write and maintain the on-disk task plan.",
        "worker": "Implement tasks and hand evidence to the planner.",
        "isolated-reviewer": "Inspect files in isolated review context.",
        "workflow-summary": "Summarize main context.",
    }
    # Summarization is a direct model call rather than a named agent. Bind its
    # identity here in the sample harness; workflow code needs no simulation logic.
    # Forty authored summary entries bound this scenario's on-demand compactions.
    summary = next(entry for entry in CONVERSATION if entry["role"] == "workflow-summary")
    scenario = [*CONVERSATION, *[dict(summary) for _ in range(39)]]
    return tuple(
        SimulatedModel(
            conversation=scenario, cache_reuse=False,
            agent_name=role if role == "workflow-summary" else None,
            metadata={"report_description": description},
        )
        for role, description in roles.items()
    )


def build_scripted_models(options):
    """Keep role-specific compaction ledgers; workflow options set actual budgets."""
    names = ("planner", "worker", "isolated-reviewer", "workflow-summary")
    return dict(zip(names, make_simulated_models(), strict=True))
