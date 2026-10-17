"""Explicit batch/LR branches preserve training state and remain resumable."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from helpers import state
from tnlm.config import Config
from tnlm.train import train_run


def _run(cfg, path, steps=1, **kwargs):
    return train_run(cfg, path, max_steps=steps, max_seconds=120, device='cpu', **kwargs)


def _equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
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


def _optimizer_scaled(actual, expected, scale):
    if 'state' not in expected:  # Muon's composite optimizer.
        assert actual.keys() == expected.keys()
        for name in expected:
            _optimizer_scaled(actual[name], expected[name], scale)
        return
    _equal(actual['state'], expected['state'])
    assert len(actual['param_groups']) == len(expected['param_groups'])
    for new, old in zip(actual['param_groups'], expected['param_groups']):
        assert set(new) == set(old) | {'base_lr'}
        assert new['base_lr'] == pytest.approx(old.get('base_lr', old['lr']) * scale, rel=1e-15)
        for key, value in old.items():
            if key in ('lr', 'base_lr'):
                assert new[key] == pytest.approx(value * scale, rel=1e-15)
            else:
                _equal(new[key], value)


@pytest.mark.parametrize('optimizer', ['adamw', 'muon'])
def test_batch_branch_preserves_initial_state_and_exact_split_resume(config, tmp_path, optimizer):
    config.train.optimizer = optimizer
    config.train.lr = .001
    config.train.aux_adamw_lr = .0003 if optimizer == 'muon' else None
    config.train.global_prediction_tokens_per_update = 1
    config.train.rung_total_tokens = 125
    config.raw['project'].update(keep_recovery_checkpoints=0, milestone_tokens=[])
    original_config = copy.deepcopy(config.to_dict())
    parent = _run(config, tmp_path / 'parent', steps=100)
    assert parent['progress']['steps'] == parent['progress']['warmup_steps'] == 100
    parent_state = state(Path(parent['checkpoint']))
    whole = _run(config, tmp_path / 'whole', steps=4, resume=parent['checkpoint'],
                 batch_tokens=7, lr_scale=.5)
    initial = state(tmp_path / 'whole' / 'initial')
    for key in ('model', 'sampler', 'rng'):
        _equal(initial[key], parent_state[key])
    _optimizer_scaled(initial['optimizer'], parent_state['optimizer'], .5)
    for key in ('steps', 'tokens', 'scheduler_step', 'warmup_steps'):
        assert initial['progress'][key] == parent_state['progress'][key]
    assert initial['config']['train']['global_prediction_tokens_per_update'] == 7
    assert initial['config']['train']['lr'] == config.train.lr * .5
    if optimizer == 'muon':
        assert initial['config']['train']['aux_adamw_lr'] == config.train.aux_adamw_lr * .5
    assert config.to_dict() == original_config  # Branch derives a copy.
    assert [row['prediction_tokens'] for row in whole['history']] == [7, 7, 7, 4]
    assert whole['progress']['tokens'] == 125
    assert whole['progress']['steps'] == 104
    first = _run(config, tmp_path / 'first', steps=2, resume=parent['checkpoint'],
                 batch_tokens=7, lr_scale=.5)
    child_path = tmp_path / 'child.yaml'
    child_path.write_text(yaml.safe_dump(initial['config']))
    child = Config.from_yaml(child_path)
    resumed = _run(child, tmp_path / 'resumed', steps=2, resume=first['checkpoint'])
    expected, actual = state(Path(whole['checkpoint'])), state(Path(resumed['checkpoint']))
    for key in ('model', 'optimizer', 'sampler', 'rng', 'progress'):
        _equal(actual[key], expected[key])
    assert [row['batch_hashes'] for row in whole['history'][2:]] == [
        row['batch_hashes'] for row in resumed['history']]
    if optimizer == 'adamw':
        longer_context = copy.deepcopy(config)
        longer_context.raw['data']['context'] = 8
        with pytest.raises(ValueError, match='context'):
            _run(longer_context, tmp_path / 'context-branch', resume=parent['checkpoint'],
                 context_fork=True, batch_tokens=7, lr_scale=.5)
        with pytest.raises(ValueError, match='cooldown'):
            _run(config, tmp_path / 'simultaneous-cooldown', resume=parent['checkpoint'],
                 cooldown_steps=2, batch_tokens=7, lr_scale=.5)
        cooldown = _run(config, tmp_path / 'cooled-parent', resume=parent['checkpoint'],
                        cooldown_steps=2)
        with pytest.raises(ValueError, match='cooldown'):
            _run(config, tmp_path / 'from-cooldown', resume=cooldown['checkpoint'],
                 batch_tokens=7, lr_scale=.5)


def test_batch_and_lr_changes_need_explicit_branch_and_completed_warmup(config, tmp_path):
    parent = _run(config, tmp_path / 'parent')
    for field, value in [('global_prediction_tokens_per_update', 14), ('lr', .02)]:
        changed = copy.deepcopy(config)
        setattr(changed.train, field, value)
        with pytest.raises(ValueError, match='resume config mismatch'):
            _run(changed, tmp_path / field, resume=parent['checkpoint'])
    with pytest.raises(ValueError, match='warmup'):
        _run(config, tmp_path / 'early', resume=parent['checkpoint'], batch_tokens=14, lr_scale=2.)


@pytest.mark.parametrize('controls', [
    {'batch_tokens': 14}, {'lr_scale': 2.},
    {'batch_tokens': 14, 'lr_scale': 2.},
])
def test_batch_branch_requires_explicit_parent_and_paired_controls(config, tmp_path, controls):
    with pytest.raises(ValueError):
        _run(config, tmp_path / 'invalid', **controls)
    assert not (tmp_path / 'invalid' / 'final').exists()


@pytest.mark.parametrize('controls', [
    {'batch_tokens': 0, 'lr_scale': 1.},
    {'batch_tokens': True, 'lr_scale': 1.},
    {'batch_tokens': 3.5, 'lr_scale': 1.},
    {'batch_tokens': 14, 'lr_scale': 0.},
    {'batch_tokens': 14, 'lr_scale': True},
    {'batch_tokens': 14, 'lr_scale': float('nan')},
    {'batch_tokens': 14, 'lr_scale': float('inf')},
])
def test_batch_branch_rejects_invalid_controls_before_loading(config, tmp_path, controls):
    with pytest.raises(ValueError):
        _run(config, tmp_path / 'invalid', resume=tmp_path / 'missing-parent', **controls)
    assert not (tmp_path / 'invalid' / 'final').exists()


def test_batch_branch_rejects_cosine_schedule(config, tmp_path):
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 1000
    with pytest.raises(ValueError, match='cosine|constant'):
        _run(config, tmp_path / 'cosine-branch', resume=tmp_path / 'missing-parent',
             batch_tokens=14, lr_scale=.5)
