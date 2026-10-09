"""Verify generated tests really execute and failed evidence blocks completion.

Real child Python processes exercise failure, empty-suite and timeout receipts.
Scripted model approvals deliberately disagree with execution to prove the graph
enforces the boundary independently of model judgment, in both client modes.
AI attribution: Generated with AI assistance by Codex.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_runtime.harness.simulated_model import ScriptedChatModel
from agent_runtime.workflows.context_budget import build_workflow
from agent_runtime.workflows.exercise_tests import run_tests


@pytest.mark.parametrize('body,status,count', [
    ('self.assertEqual(slugify("Hi"), "hi")', 'passed', 1),
    ('self.assertEqual(slugify("Hi"), "wrong")', 'failed', 1),
    ('raise RuntimeError("test failure")', 'failed', 1),
    ('self.skipTest("not tested")', 'failed', 1),
])
def test_real_suite_results(tmp_path, body, status, count):
    """Counts and approval status must reflect the process, not a model claim."""
    (tmp_path / 'slug.py').write_text('def slugify(value): return value.lower()\n')
    (tmp_path / 'test_slug.py').write_text(
        'import unittest\nfrom slug import slugify\n'
        'class Tests(unittest.TestCase):\n    def test_slug(self):\n        ' + body + '\n')
    result = run_tests(tmp_path, required=True)
    assert result['status'] == status
    assert result['tests_run'] == count
    assert 'Ran 1 test' in result['stderr']
    if status == 'passed':
        assert result['exit_code'] == 0


@pytest.mark.parametrize('source,status', [
    ('', 'failed'),
    ('import sys; sys.exit(0)', 'failed'),
    ('this is invalid python!', 'failed'),
    ('import time; time.sleep(10)', 'timeout'),
])
def test_incomplete_execution_never_passes(tmp_path, source, status):
    """Zero tests and premature exit cannot masquerade as a completed suite."""
    (tmp_path / 'test_slug.py').write_text(source)
    result = run_tests(tmp_path, required=True, timeout=0.5)
    assert result['status'] == status


def test_missing_tests_and_fresh_imports(tmp_path):
    """Allow pre-test implementation and rerun repaired files in a fresh process."""
    assert run_tests(tmp_path, required=False)['status'] == 'not_yet_written'
    assert run_tests(tmp_path, required=True)['status'] == 'missing'
    (tmp_path / 'test_slug.py').write_text(
        'import unittest\nfrom slug import answer\n'
        'class Tests(unittest.TestCase):\n'
        '    def test_answer(self): self.assertEqual(answer, 2)\n')
    (tmp_path / 'slug.py').write_text('answer = 1\n')
    assert run_tests(tmp_path, required=True)['status'] == 'failed'
    (tmp_path / 'slug.py').write_text('answer = 2\n')
    assert run_tests(tmp_path, required=True)['status'] == 'passed'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_bad_execution_overrides_model_approval(tmp_path, asynchronous):
    """A reviewer saying approve cannot let an empty test suite reach the planner."""
    (tmp_path / 'test_slug.py').write_text('')
    (tmp_path / 'plan.md').write_text('Current task: implement and verify the assigned files.\n')
    assignment = {'action': 'work', 'task_id': 'T2', 'files': ['/test_slug.py'], 'message': 'Write tests'}
    review = {'task_id': 'T2', 'verdict': 'approve', 'evidence': 'Looks correct'}
    models = [ScriptedChatModel(responses=[AIMessage(content=json.dumps(assignment))]),
              ScriptedChatModel(responses=[AIMessage(content='Tests ready')]),
              ScriptedChatModel(responses=[AIMessage(content=json.dumps(review))]),
              ScriptedChatModel(responses=[AIMessage(content='Summary')])]
    graph = build_workflow(*models, workspace_dir=tmp_path, max_attempts=1)
    payload = {'messages': [HumanMessage('Create tests')]}
    result = asyncio.run(graph.ainvoke(payload)) if asynchronous else graph.invoke(payload)
    assert result['outcome'] == 'review_limit'
    assert result['review']['verdict'] == 'revise'
    assert result['test_execution']['status'] == 'failed'
    assert result['completed'] == []
    assert any('Observed test execution:' in m.text for m in result['messages'])


@pytest.mark.parametrize('asynchronous', [False, True])
def test_failed_tests_are_rerun_after_worker_repair(tmp_path, asynchronous):
    """The repair loop must replace failed evidence before allowing completion."""
    (tmp_path / 'slug.py').write_text('answer = 1\n')
    (tmp_path / 'test_slug.py').write_text(
        'import unittest\nfrom slug import answer\nclass Tests(unittest.TestCase):\n'
        '    def test_answer(self): self.assertEqual(answer, 2)\n')
    (tmp_path / 'plan.md').write_text('Current task: implement and verify the assigned files.\n')
    assignment = {'action': 'work', 'task_id': 'T1', 'files': ['/slug.py'], 'message': 'Fix answer'}
    done = {'action': 'finish', 'task_id': '', 'files': [], 'message': 'Verified'}
    approval = AIMessage(content=json.dumps({'task_id': 'T1', 'verdict': 'approve',
                                            'evidence': 'Source reviewed'}))
    models = [ScriptedChatModel(responses=[AIMessage(content=json.dumps(x))
                                          for x in [assignment, done]]),
              ScriptedChatModel(responses=[
                  AIMessage(content='First candidate'),
                  AIMessage(content='', tool_calls=[{
                      'id': 'repair', 'name': 'edit_file',
                      'args': {'file_path': '/slug.py', 'old_string': 'answer = 1',
                               'new_string': 'answer = 2'}}]),
                  AIMessage(content='Repaired')]),
              ScriptedChatModel(responses=[approval, approval]),
              ScriptedChatModel(responses=[AIMessage(content='Summary')])]
    graph = build_workflow(*models, workspace_dir=tmp_path, max_attempts=2)
    payload = {'messages': [HumanMessage('Fix answer')]}
    result = asyncio.run(graph.ainvoke(payload)) if asynchronous else graph.invoke(payload)
    receipts = [json.loads(m.text.removeprefix('Observed test execution: '))
                for m in result['messages'] if m.text.startswith('Observed test execution: ')]
    assert [r['status'] for r in receipts] == ['failed', 'passed']
    assert [r['attempt'] for r in receipts] == [1, 2]
    assert result['completed'] == ['T1']
    assert result['outcome'] == 'complete'
    assert result['test_execution']['exit_code'] == 0


def test_planner_creates_missing_plan_before_dispatch(tmp_path):
    """Recover Sol's observed premature dispatch in the role that owns the plan."""
    assignment = {'action': 'work', 'task_id': 'T1', 'files': ['/slug.py'], 'message': 'Implement'}
    done = {'action': 'finish', 'task_id': '', 'files': [], 'message': 'Reviewed'}
    planning = ScriptedChatModel(responses=[
        AIMessage(content=json.dumps(assignment)),
        AIMessage(content='', tool_calls=[{'id': 'plan', 'name': 'write_file', 'args': {
            'file_path': '/plan.md', 'content': 'T1: implement slug.py. Tests follow in T2.'}}]),
        AIMessage(content=json.dumps(assignment)), AIMessage(content=json.dumps(done))])
    models = [planning, ScriptedChatModel(responses=[AIMessage(content='Implemented')]),
              ScriptedChatModel(responses=[AIMessage(content=json.dumps({
                  'task_id': 'T1', 'verdict': 'approve', 'evidence': 'Source reviewed'}))]),
              ScriptedChatModel(responses=[AIMessage(content='Summary')])]
    result = build_workflow(*models, workspace_dir=tmp_path).invoke(
        {'messages': [HumanMessage('Implement slugify')]})
    assert (tmp_path / 'plan.md').is_file()
    assert result['completed'] == ['T1']
    assert any('Write /plan.md' in m.text for m in result['messages'])
