"""Verify model-list dispatch and offline comparison without paid provider calls.

Each stub child writes a real normalized bundle, allowing the ordinary renderer
and selection validation to run. Existing evidence must survive preflight errors.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qa_helpers import rubric_definition

from reporting.pricing import Prices
from reporting.schema import QAEvaluation, Run, Step, Usage

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['openai:gpt-5.5', 'codex:gpt-5.5', 'openai:gpt-5.6-luna',
          'codex:gpt-5.6-luna', 'codex:gpt-6-sol']


@pytest.mark.parametrize('reuse_rubric', [False, True])
def test_five_models_share_configuration_and_saved_only_is_offline(tmp_path, monkeypatch, reuse_rubric):
    """The caller supplies one list; only provider/model differ between launches."""
    planner = Mock(side_effect=lambda values, goal, title: rubric_definition(goal))
    monkeypatch.setattr("agent_runtime.harness.qa_judge.create_rubric", planner)
    main = runpy.run_path(str(ROOT / 'scripts/run_model_comparison.py'))['main']
    calls = []
    monkeypatch.setenv('LG_MAX_TOKENS', '1024')

    def child(command, **kwargs):
        """Produce tiny receipts using the exact model selected for the child."""
        calls.append((command, kwargs))
        env = kwargs['env']
        directory = Path(command[command.index('--out') + 1])
        run = Run(id='run', title='Fixture', status='ok', steps=[
            Step(id='model', name='Assistant', kind='model', start_ns=0,
                 end_ns=1_000_000_000, status='ok', provider=env['LG_PROVIDER'],
                 model=env['LG_MODEL'], effort=env['LG_EFFORT'], context={'report_turn': 1},
                 usage=Usage(input_tokens=10, output_tokens=10))])
        (directory / 'run.json').write_text(run.model_dump_json())
        (directory / 'prices.json').write_text(Prices(as_of='2026-10-06', note='Fixture', models={}).model_dump_json())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', child)
    args = ['--models', *MODELS, '--qa', '--out', str(tmp_path)]
    if reuse_rubric:
        # A subset rerun uses the existing comparison's exact criteria, without
        # paying for another planner or silently changing score meanings.
        from agent_runtime.harness.sample_catalog import SampleCatalog
        sample = SampleCatalog().get('simple_chat')
        supplied = tmp_path / 'supplied-rubric.json'
        supplied.write_text(rubric_definition(sample.goal).model_dump_json())
        args.extend(['--qa-rubric', str(supplied)])
    assert main(args) == 0
    assert len(calls) == 5
    for selection, (command, options) in zip(MODELS, calls):
        env = options['env']
        assert f"{env['LG_PROVIDER']}:{env['LG_MODEL']}" == selection
        assert env['LG_EFFORT'] == 'medium' and env['LG_AVAILABLE_MODELS'] == ''
        assert env['LG_USER_EFFORT'] == env['LG_QA_EFFORT'] == 'high'
        assert '--qa' in command and '--live' in command
        assert command[command.index('--client') + 1] == 'agent'
        assert options['stdin'] == subprocess.DEVNULL
        assert env['LG_MAX_TOKENS'] == ''
        assert Path(env['LG_QA_RUBRIC']) == tmp_path / 'qa-rubric.json'
    before = {p: p.read_bytes() for p in tmp_path.glob('*/*.json')}
    calls.clear()
    assert main([*args, '--saved-only']) == 0
    assert not calls and all(p.read_bytes() == data for p, data in before.items())
    if reuse_rubric:
        planner.assert_not_called()
        assert (tmp_path / 'qa-rubric.json').read_text() == rubric_definition(sample.goal).model_dump_json(indent=2)
    else:
        planner.assert_called_once()
    assert 'codex:gpt-6-sol' in (tmp_path / 'report.html').read_text()
    with pytest.raises(SystemExit):
        main(args)
    assert not calls and all(p.read_bytes() == data for p, data in before.items())


def test_failed_child_does_not_publish_comparison(tmp_path, monkeypatch):
    """Failure evidence is retained while later trials and report publication stop."""
    monkeypatch.setattr("agent_runtime.harness.qa_judge.create_rubric",
                        lambda values, goal, title: rubric_definition(goal))
    main = runpy.run_path(str(ROOT / 'scripts/run_model_comparison.py'))['main']
    calls = []

    def fail(command, **kwargs):
        """Exercise exit handling independently of any provider implementation."""
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(SystemExit):
        main(['--models', *MODELS, '--out', str(tmp_path)])
    assert len(calls) == 1 and not (tmp_path / 'report.html').exists()
    assert (tmp_path / 'gpt-5.5/run.log').exists()


def test_supplied_rubric_with_wrong_goal_stops_before_paid_work(tmp_path, monkeypatch, capsys):
    """Reusing criteria cannot quietly score a different task or buy a repair."""
    supplied = tmp_path / 'wrong-goal.json'
    supplied.write_text(rubric_definition('An unrelated goal').model_dump_json())
    planner = Mock(side_effect=AssertionError('No rubric planning expected'))
    child = Mock(side_effect=AssertionError('No Agent execution expected'))
    monkeypatch.setattr('agent_runtime.harness.qa_judge.create_rubric', planner)
    monkeypatch.setattr(subprocess, 'run', child)
    main = runpy.run_path(str(ROOT / 'scripts/run_model_comparison.py'))['main']
    with pytest.raises(SystemExit) as error:
        main(['--models', *MODELS, '--qa', '--qa-rubric', str(supplied),
              '--out', str(tmp_path / 'results')])
    assert error.value.code == 1
    assert 'Supplied rubric goal' in capsys.readouterr().err
    planner.assert_not_called()
    child.assert_not_called()


def test_rescore_reuses_execution_and_one_rubric(tmp_path, monkeypatch):
    """Rescoring may replace QA but cannot invoke Agents or edit their receipts."""
    goal = 'Explain caching'
    planner = Mock(return_value=rubric_definition(goal))
    monkeypatch.setattr('agent_runtime.harness.qa_judge.create_rubric', planner)
    monkeypatch.setattr('agent_runtime.harness.sample_catalog.SampleCatalog',
                        lambda: SimpleNamespace(get=lambda _: SimpleNamespace(goal=goal, name='Task')))
    seen = []
    def judge(values, *, goal, rubric):
        """Record the contract passed to each independent assessment."""
        seen.append(rubric.fingerprint)
        return lambda run, prices: QAEvaluation(status='completed', provider='fixture',
                                                model='judge', goal=goal, rubric=rubric)
    monkeypatch.setattr('agent_runtime.harness.qa_judge.QAJudge', judge)
    monkeypatch.setattr(subprocess, 'run', Mock(side_effect=AssertionError('Agent must not rerun')))
    main = runpy.run_path(str(ROOT / 'scripts/run_model_comparison.py'))['main']
    originals = {}
    for model in ['first', 'second']:
        directory = tmp_path / model
        directory.mkdir()
        run = Run(id=model, title='Task', status='ok', steps=[
            Step(id='answer', name='Agent', kind='model', status='ok', start_ns=0,
                 end_ns=1_000_000_000, provider='openai', model=model, effort='medium')],
            qa=QAEvaluation(status='completed', provider='fixture', model='judge', goal=goal))
        originals[model] = run.steps
        (directory / 'run.json').write_text(run.model_dump_json())
        (directory / 'prices.json').write_text(Prices(as_of='2026-10-06', note='Fixture', models={}).model_dump_json())
        (directory / 'spans.jsonl').write_text('original evidence')
    assert main(['--models', 'openai:first', 'openai:second', '--rescore', '--out', str(tmp_path)]) == 0
    planner.assert_called_once()
    assert len(seen) == 2 and len(set(seen)) == 1
    for model, steps in originals.items():
        assert Run.model_validate_json((tmp_path / model / 'run.json').read_text()).steps == steps
        assert (tmp_path / model / 'spans.jsonl').read_text() == 'original evidence'


@pytest.mark.parametrize('models', [['codex:'], ['unknown:model', 'codex:model'],
                                    ['codex:gpt-5.5', 'codex:gpt-5.5']])
def test_invalid_model_lists_fail_before_execution(models):
    """Invalid selections cannot start a partially configured experiment."""
    main = runpy.run_path(str(ROOT / 'scripts/run_model_comparison.py'))['main']
    with pytest.raises(SystemExit):
        main(['--models', *models])
