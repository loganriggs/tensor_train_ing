"""CPU training evidence for opt-in disposable screening checkpoints."""
import copy
from pathlib import Path

import pytest
import torch

import tnlm.checkpoint as checkpoint_module
from helpers import state
from tnlm.train import train_run


def _run(config, path, steps=3, **kwargs):
    return train_run(config, path, max_steps=steps, max_seconds=60, device='cpu', **kwargs)


def _equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize('save_initial', [False, True])
def test_zero_recovery_skips_writes_even_with_step_interval_one(config, tmp_path, monkeypatch, save_initial):
    config.raw['project'].update(keep_recovery_checkpoints=0,
        save_initial_checkpoint=save_initial, checkpoint_every_steps=1, milestone_tokens=[])
    calls = []
    original = checkpoint_module.save_checkpoint

    def record(path, *args, **kwargs):
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module, 'save_checkpoint', record)
    run = tmp_path / 'zero'
    result = _run(config, run)
    assert result['progress']['steps'] == 3
    assert calls == (['initial', 'final'] if save_initial else ['final'])
    assert (run / 'initial').exists() is save_initial
    assert not list(run.glob('recovery-*'))
    assert state(run / 'final')['progress']['steps'] == 3


@pytest.mark.parametrize('keep', [1, 2])
def test_recovery_replacement_prunes_only_after_success(config, tmp_path, monkeypatch, keep):
    # Leaving keep=2 and the initial-save field absent verifies the old defaults.
    config.raw['project'].update(checkpoint_every_steps=1, milestone_tokens=[])
    if keep == 1:
        config.raw['project']['keep_recovery_checkpoints'] = keep
    original = checkpoint_module.save_checkpoint
    observed = []

    def inspect_replacement(path, *args, **kwargs):
        path = Path(path)
        previous = sorted(path.parent.glob('recovery-*'))
        answer = original(path, *args, **kwargs)
        if path.name.startswith('recovery-'):
            assert all((old / 'manifest.json').is_file() for old in previous)
            assert (path / 'manifest.json').is_file()
            observed.append(len(previous))
        return answer

    monkeypatch.setattr(checkpoint_module, 'save_checkpoint', inspect_replacement)
    run = tmp_path / 'rolling'
    _run(config, run)
    assert observed == [0, 1, min(keep, 2)]
    assert sorted(path.name for path in run.glob('recovery-*')) == [
        f'recovery-{step:08d}' for step in range(4 - keep, 4)]
    assert (run / 'initial' / 'manifest.json').is_file()
    assert state(run / 'final')['progress']['steps'] == 3


def test_screening_keeps_final_and_mandatory_branch_without_ordinary_milestones(config, tmp_path):
    config.train.rung_total_tokens = 21
    config.raw['project'].update(keep_recovery_checkpoints=0, save_initial_checkpoint=False,
        checkpoint_every_steps=1, milestone_tokens=[])
    run = tmp_path / 'branch'
    result = _run(config, run)
    branch = run / 'milestone-000000000018'
    assert branch.is_dir()
    assert result['progress']['tokens'] == 21
    _equal(state(branch)['model'], state(run / 'final')['model'])
    assert sorted(path.name for path in run.iterdir() if path.is_dir()) == [
        'final', 'milestone-000000000018']


def test_screening_resume_matches_default_retention_training(config, tmp_path):
    config.raw['project'].update(checkpoint_every_steps=1, milestone_tokens=[])
    _run(config, tmp_path / 'default')
    screening = copy.deepcopy(config)
    screening.raw['project'].update(keep_recovery_checkpoints=0, save_initial_checkpoint=False)
    prefix = _run(screening, tmp_path / 'prefix', steps=1)
    _run(screening, tmp_path / 'resumed', steps=2, resume=prefix['checkpoint'])
    expected = state(tmp_path / 'default' / 'final')
    actual = state(tmp_path / 'resumed' / 'final')
    for field in ('model', 'optimizer', 'sampler', 'progress'):
        _equal(actual[field], expected[field])
    assert not (tmp_path / 'resumed' / 'initial').exists()
    assert not list((tmp_path / 'resumed').glob('recovery-*'))


def test_failed_recovery_replacement_keeps_previous_committed_state(config, tmp_path, monkeypatch):
    config.raw['project'].update(keep_recovery_checkpoints=1, save_initial_checkpoint=False,
        checkpoint_every_steps=1, milestone_tokens=[])
    original = checkpoint_module.save_checkpoint

    def fail_second(path, *args, **kwargs):
        if Path(path).name == 'recovery-00000002':
            raise OSError('injected recovery write failure')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module, 'save_checkpoint', fail_second)
    run = tmp_path / 'failed'
    with pytest.raises(OSError, match='injected recovery write failure'):
        _run(config, run)
    assert state(run / 'recovery-00000001')['progress']['steps'] == 1
    assert (run / 'recovery-00000001' / 'manifest.json').is_file()
    assert not (run / 'recovery-00000002').exists()
    assert not (run / 'initial').exists()
    assert not (run / 'final').exists()
