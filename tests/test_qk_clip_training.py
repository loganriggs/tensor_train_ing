"""CPU evidence for fixed-probe QK clipping in actual serial training."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import state
from tnlm.data import PackedSampler, batch_hash
from tnlm.normalization import lagged_modules
from tnlm.train import train_run


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


def _run(config, path, steps=1, **kwargs):
    return train_run(config, path, max_steps=steps, max_seconds=60, device='cpu', **kwargs)


@pytest.mark.parametrize('normalization', ['pre_rms', 'lagged'])
def test_qk_clip_real_training_and_global_cadence_resume(config, tmp_path, monkeypatch, normalization):
    import tnlm.train as trainer
    import tnlm.qk_clip as clipping

    config.model.normalization = normalization
    config.model.final_normalization = 'lagged' if normalization == 'lagged' else 'rms'
    config.train.qk_clip_tau = .001
    config.train.qk_clip_every_steps = 2
    config.train.stability_every_steps = 0
    config.raw['project'].update(keep_recovery_checkpoints=0, save_initial_checkpoint=False,
                                 milestone_tokens=[])
    optimizers, invocations = [], []
    original_optimizer, original_clip = trainer.make_optimizer, clipping.qk_clip_

    def capture_optimizer(*args, **kwargs):
        result = original_optimizer(*args, **kwargs)
        optimizers.append(result)
        return result

    def checked_clip(model, idx, doc, tau):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert all(not norm.collecting for norm in lagged_modules(model))
        before = copy.deepcopy(optimizers[-1].state_dict())
        result = original_clip(model, idx, doc, tau)
        _equal(optimizers[-1].state_dict(), before)
        invocations.append(result)
        return result

    monkeypatch.setattr(trainer, 'make_optimizer', capture_optimizer)
    monkeypatch.setattr(clipping, 'qk_clip_', checked_clip)
    whole = _run(config, tmp_path / 'whole', steps=4)
    prefix = _run(config, tmp_path / 'prefix', steps=1)
    resumed = _run(config, tmp_path / 'resumed', steps=3, resume=prefix['checkpoint'])
    expected, actual = state(Path(whole['checkpoint'])), state(Path(resumed['checkpoint']))
    for key in ('model', 'optimizer', 'sampler', 'progress', 'rng'):
        _equal(actual[key], expected[key])
    assert len(invocations) == 4
    assert all(row['clipped_heads'] > 0 and row['max_before'] > .001 for row in invocations)
    assert expected['config']['train']['qk_clip_tau'] == .001
    assert expected['config']['train']['qk_clip_every_steps'] == 2
    whole_rows = [json.loads(line) for line in (tmp_path / 'whole/metrics.jsonl').read_text().splitlines()]
    split_rows = [json.loads(line) for line in (tmp_path / 'resumed/metrics.jsonl').read_text().splitlines()]
    clips = [row for row in whole_rows if 'qk_clip' in row]
    split_clips = [row for row in split_rows if 'qk_clip' in row]
    assert [row['step'] for row in clips] == [2, 4]
    assert [row['step'] for row in split_clips] == [2, 4]
    assert not any('stability' in row for row in whole_rows)
    sampler = PackedSampler(config.raw['data']['manifest'], split='train', batch_size=1,
                            context=4, seed=config.train.seed)
    identity = clips[0]['qk_clip']['probe']
    assert identity['batch_sha256'] == batch_hash(sampler.next_batch())
    assert identity['manifest_sha256'] == sampler.manifest_hash
    assert identity['split'] == 'train'
    for left, right in zip(clips, split_clips):
        assert left['qk_clip']['probe'] == right['qk_clip']['probe'] == identity
        assert left['qk_clip']['seconds'] > 0
        for key in ('clipped_heads', 'max_before', 'factors', 'tau'):
            assert left['qk_clip'][key] == right['qk_clip'][key]


def test_qk_clip_precedes_stability_diagnostic(config, tmp_path):
    config.train.qk_clip_tau = .001
    config.train.qk_clip_every_steps = 1
    config.train.stability_every_steps = 1
    result = _run(config, tmp_path / 'diagnostic')
    row = result['history'][0]
    assert row['qk_clip']['clipped_heads'] > 0
    assert row['qk_clip']['probe'] == row['stability']['probe']
    for layer in row['stability']['stats']['per_layer']:
        assert layer['S1_max_abs'] <= .001 * 1.001
        if 'S2_max_abs' in layer:
            assert layer['S2_max_abs'] <= .001 * 1.001


def test_qk_clip_default_off_constructs_no_probe(config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default-off training must not construct a probe')
    monkeypatch.setattr('tnlm.train.TrainingStabilityProbe', forbidden)
    result = _run(config, tmp_path / 'default')
    assert all('qk_clip' not in row for row in result['history'])


@pytest.mark.parametrize('tau,cadence', [
    (-1., 1), (True, 1), (float('nan'), 1), (float('inf'), 1), ('1', 1),
    (.001, 0), (.001, -1), (.001, True), (.001, 1.5), (.001, '1'),
])
def test_qk_clip_invalid_policy_rejected_before_output(config, tmp_path, tau, cadence):
    config.train.qk_clip_tau = tau
    config.train.qk_clip_every_steps = cadence
    with pytest.raises(ValueError):
        _run(config, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_qk_clip_rejects_spectral_before_output(config, tmp_path):
    config.model.normalization = 'spectral_rms'
    config.train.qk_clip_tau = .001
    config.train.qk_clip_every_steps = 2
    with pytest.raises((ValueError, NotImplementedError), match='spectral'):
        _run(config, tmp_path / 'spectral')
    assert not (tmp_path / 'spectral').exists()


def test_resume_cannot_enable_qk_clip_silently(config, tmp_path):
    parent = _run(config, tmp_path / 'parent')
    config.train.qk_clip_tau = .001
    config.train.qk_clip_every_steps = 2
    with pytest.raises(ValueError, match='resume config mismatch'):
        _run(config, tmp_path / 'changed', resume=parent['checkpoint'])
    assert not (tmp_path / 'changed' / 'final').exists()
