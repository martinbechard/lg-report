"""Prove that Agent wall time includes workflow work without double counting.

Synthetic traces distinguish tools and retries from user waits, judge overhead,
parallel branches, and missing boundaries. Token usage is deliberately absent:
wall time must not depend on how much text a model produces.
AI assistance: Codex. Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import pytest

from reporting.performance import agent_elapsed_seconds
from reporting.schema import Run, Step


def span(name, start, end, *, parent=None, kind='workflow', turn=1, role=None, status='ok'):
    """Build nanosecond trace spans with explicit ownership and turn attribution."""
    context = {'report_turn': turn} if turn is not None else {}
    if role:
        context['model_role'] = role
    return Step(id=name, name=name, kind=kind, parent_id=parent,
                start_ns=int(start * 1e9), end_ns=int(end * 1e9), status=status, context=context)


def test_complete_turn_time_includes_tools_tests_and_retries_excludes_gaps():
    """Model-only sums omit work; whole-run duration incorrectly includes waits."""
    run = Run(id='run', title='Time', status='ok', steps=[
        span('turn1', 0, 10),
        span('model1', 1, 3, parent='turn1', kind='model', status='error'),
        span('retry', 3, 5, parent='turn1', kind='model'),
        span('tests', 5, 9, parent='turn1', kind='tool'),
        span('user', 10, 30, kind='model', role='user'),
        span('turn2', 40, 45, turn=2),
        span('model2', 41, 42, parent='turn2', kind='model', turn=2),
        span('judge', 45, 100, kind='model', role='qa'),
        span('outcome', 100, 100, turn=None),
    ])
    assert agent_elapsed_seconds(run) == 15


def test_parallel_roots_nested_spans_and_excluded_roles_count_once():
    """Union Agent intervals and subtract the union of nested evaluation work."""
    run = Run(id='run', title='Time', status='ok', steps=[
        span('root1', 0, 10), span('root2', 5, 15, turn=2),
        span('a', 1, 8, parent='root1', kind='model'),
        span('b', 6, 14, parent='root2', kind='model', turn=2),
        span('user', 2, 4, parent='root1', role='user'),
        span('user-model', 2.5, 3.5, parent='user', role='user', kind='model'),
        span('qa', 3, 6, parent='root1', role='qa', kind='model'),
    ])
    assert agent_elapsed_seconds(run) == 11  # [0,15] minus [2,6], not sum of spans.


@pytest.mark.parametrize('change', ['missing_turn', 'missing_root', 'incomplete', 'outside_root', 'failed', 'static'])
def test_incomplete_execution_boundaries_are_not_partial_speed_scores(change):
    """Unknown or invalid boundary evidence must not look like a faster run."""
    run = Run(id='run', title='Time', status='ok', steps=[
        span('root', 0, 10), span('model', 1, 8, parent='root', kind='model'),
    ])
    if change == 'missing_turn': run.steps[0].context = {}
    elif change == 'missing_root': run.steps[1].parent_id = None; run.steps[1].context = {}
    elif change == 'incomplete': run.steps[0].status = 'incomplete'
    elif change == 'outside_root': run.steps[1].end_ns = int(11e9)
    elif change == 'failed': run.status = 'error'
    elif change == 'static': run.demo = True
    assert agent_elapsed_seconds(run) is None


def test_direct_model_turn_is_a_valid_complete_boundary():
    """Simple direct-model invocations still have an observable Agent duration."""
    run = Run(id='run', title='Time', status='ok', steps=[span('model', 1, 3, kind='model')])
    assert agent_elapsed_seconds(run) == 2
