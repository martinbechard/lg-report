"""Script file-tool decisions for the context-budget planning exercise.

Real native tools write and read the plan, code, and tests. Scripted model
responses make the lesson deterministic; compaction remains native middleware
behavior, and reported usage is illustrative rather than provider billing.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json

from langchain_core.messages import AIMessage

from agent_runtime.harness.demo_meter import ContextSimulation
from agent_runtime.harness.simulated_model import MeteredDemoModel


class CompactingDemoModel(MeteredDemoModel):
    """Meter each rewritten request without claiming cache reuse."""

    def _generate(self, messages, *args, **kwargs):
        # The generic demo meter assumes append-only history. Native compaction
        # replaces old messages, so its local prefix ledger must restart.
        self._simulation = ContextSimulation()
        result = super()._generate(messages, *args, **kwargs)
        result.generations[0].message.response_metadata["usage_basis"] = (
            "Simulated canonical-JSON tokens; rewritten context, no cache reuse assumed"
        )
        return result


# The first request carries deliberate older context; the plan and skills are
# the durable, larger load on subsequent calls. The second turn shows rereads
# after conversation compaction without repeating the original user request.
USER_PROMPTS = [
    "Build a small, well-tested Python slug function. Make a detailed file plan first. "
    + "The implementation should be inspectable and independent of services. " * 100,
    "Continue the plan: strengthen the boundary tests, get an independent review, and record the outcome.",
]
PLAN = """# Slug utility delivery plan

Goal: provide one deterministic Python function named slugify in /slug.py. It turns a title into a lowercase ASCII slug suitable for a local document name. The sample must demonstrate file-backed coordination and independent review, not a production packaging workflow. /test_slug.py contains focused unittest coverage. Each step follows worker implementation, independent approval, then planner completion before the next step starts.

Working agreement: every task has a stable ID. An edit changes only the named task status and evidence. Pending means no artifact is claimed. Complete means the current version was independently approved and the planner recorded that approval. Blocked means a concrete missing input or failed operation. A reviewer report is evidence of inspection, not evidence that tests executed. The reviewer has read-only access and a separate context; its assignment must include all paths and intended behavior. The planner retains ownership of the plan. The worker reports implementation evidence but does not edit the plan.

Design contract: accept a string title. Normalize accented Latin letters to their ASCII base where Unicode decomposition provides one. Convert to lowercase. Treat runs of punctuation and whitespace as a single hyphen. Strip leading and trailing hyphens. Empty or separator-only text returns an empty string. A non-string input raises TypeError with a clear message. Do not use network calls, environment configuration, or generated identifiers. Preserve deterministic output for the same input.

Implementation notes: prefer unicodedata.normalize over a third-party transliteration package. Encode decomposed text as ASCII while dropping characters that have no ASCII form. Explain that limitation in a docstring because some scripts become empty. Use a regular expression for separator runs and make its pattern explicit. Keep the public function side-effect free. Avoid broad exception handlers and hidden fallback behavior. Include a purpose header, AI attribution, and the project copyright in each Python file.

Verification agreement: tests use only Python's standard library unittest. They should verify a normal title, repeated whitespace and punctuation, accented Latin input, leading and trailing separators, empty input, and invalid type. Tests are source artifacts until a command actually executes them; the agent in this workflow has file tools but no shell tool. The final report must distinguish written tests from passing tests. Review findings should name a triggering example and the affected task.

Task T1 — implement the public function. Status: pending. Acceptance: /slug.py exposes slugify; ordinary words become lowercase with one hyphen; accents with Latin decompositions lose marks; punctuation runs do not create repeated hyphens; non-string values fail visibly. Read the plan after recording completion.

Task T2 — create focused tests. Status: pending. Acceptance: /test_slug.py imports slugify and uses unittest with examples for normal titles, spacing, punctuation, accents, empty input, and invalid input. The tests should distinguish a plausible broken implementation from the expected contract. Read the plan after recording completion.

Risk notes: file reads may be paginated, so inspect the rest if a native read is truncated. A successful write does not prove behavior; it only proves content was saved. A summary may omit exact plan lines, so the current file controls status after compaction. If a tool reports an error, keep the task pending or blocked and explain why. Avoid broadening the task to packaging, CLI behavior, or external deployment. The intended learning is the relationship among plan state, conversation state, reviewer isolation, and context limits.
"""
CODE = '''"""Create deterministic ASCII slugs for local document titles.

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
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
'''
TESTS = '''"""Check the public slug contract with standard-library unittest.

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
'''
EXTRA_TESTS = '''
    def test_only_separators(self):
        self.assertEqual(slugify("...  ---"), "")

    def test_outer_punctuation(self):
        self.assertEqual(slugify("!! Great Day ??"), "great-day")
'''

PEER_BRIEF = "Implement T1 only: write /slug.py according to /plan.md. Tests belong to the next step."
WORKFLOW_SUMMARY = "The plan lives in /plan.md. Planner owns only plan updates; worker owns source and tests; reviewer independently approves each current task. The workflow supplies the authoritative current assignment and verdict after this summary. Never infer approval or the next actor from this summary."
FINAL_ANSWER = "All assigned steps were independently approved and recorded complete. Tests were written but not executed."
BOUNDARY_TASK = "Task T3 — extend boundary coverage. Status: pending. Acceptance: add separator-only and leading/trailing punctuation tests to /test_slug.py."


def call(name, args, number):
    """Build one authored request; native tools supply the actual outcome."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}-{number}"}])


def _model(responses, description):
    """Use independent response cursors with explicitly illustrative usage."""
    return CompactingDemoModel(responses=responses, metadata={"report_description": description})


def make_simulated_models():
    """Exercise every step and a rejected candidate through the real graph.

    T1 deliberately misses boundary trimming. Its reviewer rejects that version;
    the worker repairs it and a fresh review approves before any completion edit.
    T2 runs next. The second user turn adds T3 without losing earlier evidence.
    """
    def dispatch(task_id, message):
        """Author the planner decision consumed by the conditional graph edge."""
        return AIMessage(content=json.dumps({"action": "work" if task_id else "finish", "task_id": task_id, "files": (["/slug.py"] if task_id == "T1" else ["/test_slug.py"]) if task_id else [], "message": message}))

    def verdict(task_id, decision, evidence):
        """Bind the scripted assessment to the current task and inspected version."""
        return AIMessage(content=json.dumps({"task_id": task_id, "verdict": decision, "evidence": evidence}))

    def complete(task_id, title, number):
        """Record one approval, then reread before selecting the next task."""
        # Every completion is a narrow edit after a matching independent verdict.
        return [
            call("read_file", {"file_path": "/plan.md"}, number),
            call("edit_file", {"file_path": "/plan.md", "old_string": f"Task {task_id} — {title}. Status: pending.", "new_string": f"Task {task_id} — {title}. Status: complete. Review evidence: independent reviewer approved current source; tests not executed."}, number + 1),
            call("read_file", {"file_path": "/plan.md"}, number + 2),
        ]

    planner_responses = [
        call("write_file", {"file_path": "/plan.md", "content": PLAN}, 1),
        dispatch("T1", PEER_BRIEF),
        *complete("T1", "implement the public function", 2),
        dispatch("T2", "Implement T2 only: write /test_slug.py with focused unittest coverage of the plan contract. Read current /slug.py first."),
        *complete("T2", "create focused tests", 5),
        dispatch("", "T1 and T2 independently approved and complete. Tests not executed."),
        call("read_file", {"file_path": "/plan.md"}, 8),
        call("edit_file", {"file_path": "/plan.md", "old_string": "Risk notes:", "new_string": BOUNDARY_TASK + "\n\nRisk notes:"}, 9),
        dispatch("T3", "Implement T3 only: extend /test_slug.py with separator-only and outer-punctuation tests. Preserve existing tests."),
        *complete("T3", "extend boundary coverage", 10),
        dispatch("", FINAL_ANSWER),
    ]
    worker_responses = [
        call("read_file", {"file_path": "/plan.md"}, 20),
        call("write_file", {"file_path": "/slug.py", "content": CODE.replace('.strip("-")', '')}, 21),
        AIMessage(content="T1 candidate written; source review required. Tests not executed."),
        # A revise verdict returns the SAME task to the worker, not the planner.
        call("read_file", {"file_path": "/slug.py"}, 22),
        call("edit_file", {"file_path": "/slug.py", "old_string": 'ascii_title.lower())', "new_string": 'ascii_title.lower()).strip("-")'}, 23),
        AIMessage(content="T1 repaired: boundary hyphens are stripped. Please review the current file."),
        call("read_file", {"file_path": "/plan.md"}, 24),
        call("read_file", {"file_path": "/slug.py"}, 25),
        call("write_file", {"file_path": "/test_slug.py", "content": TESTS}, 26),
        AIMessage(content="T2 test source written; tests not executed. Review required."),
        call("read_file", {"file_path": "/plan.md"}, 27),
        call("read_file", {"file_path": "/test_slug.py"}, 28),
        call("edit_file", {"file_path": "/test_slug.py", "old_string": "    def test_invalid_input(self):", "new_string": EXTRA_TESTS + "\n    def test_invalid_input(self):"}, 29),
        AIMessage(content="T3 boundary cases added; tests not executed. Review required."),
    ]
    reviewer_responses = []
    for index, (task_id, decision, evidence) in enumerate([
        ("T1", "revise", 'slugify("! Hi !") returns "-hi-". Strip boundary hyphens in /slug.py. Tests not executed.'),
        ("T1", "approve", '/slug.py now strips boundary hyphens and satisfies T1. Source inspection only; tests not executed.'),
        ("T2", "approve", '/test_slug.py covers normal input, punctuation, accents, empty input and invalid types as T2 requires. Tests not executed.'),
        ("T3", "approve", '/test_slug.py adds test_only_separators and test_outer_punctuation; existing tests preserved. Tests not executed.'),
    ]):
        reviewer_responses.extend([
            call("read_file", {"file_path": "/plan.md"}, 40 + index * 3),
            call("read_file", {"file_path": "/slug.py"}, 41 + index * 3),
        ])
        if task_id != "T1":
            reviewer_responses.append(call("read_file", {"file_path": "/test_slug.py"}, 42 + index * 3))
        reviewer_responses.append(verdict(task_id, decision, evidence))
    return (
        _model(planner_responses, "Write and maintain the on-disk task plan."),
        _model(worker_responses, "Implement tasks and hand evidence to the planner."),
        _model(reviewer_responses, "Inspect files in isolated review context."),
        _model([AIMessage(content=WORKFLOW_SUMMARY)] * 40, "Summarize main context."),
    )


def build_models(options):
    """Map catalog caller names to fresh scripted adapters for one run."""
    names = ("planner", "worker", "isolated-reviewer", "workflow-summary")
    return dict(zip(names, make_simulated_models(), strict=True))
