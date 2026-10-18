"""Whole-batch score bounds, last-microbatch clipping, and serial replay."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from helpers import state
from tnlm.config import Config
from tnlm.model import TensorLM
from tnlm.qk_clip import score_bounds, set_score_bound_recording
from tnlm.runtime import full_config, resume_validator
from tnlm.train import train_run
from tnlm.training_stability import TrainingStabilityProbe


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
def test_real_batch_bound_clipping_exact_resume_without_probe(config, tmp_path, monkeypatch, normalization):
    config.model.normalization = normalization
    config.model.final_normalization = 'lagged' if normalization == 'lagged' else 'rms'
    config.train.qk_clip_source = 'batch_bound'
    config.train.qk_clip_tau = .001
    config.train.qk_clip_every_steps = 2
    config.train.stability_every_steps = 0
    config.raw['project'].update(keep_recovery_checkpoints=0, save_initial_checkpoint=False, milestone_tokens=[])

    def forbidden(*args, **kwargs):
        raise AssertionError('batch-bound mode must not allocate an independent probe')

    monkeypatch.setattr('tnlm.train.TrainingStabilityProbe', forbidden)
    whole = _run(config, tmp_path / 'whole', steps=4)
    first = _run(config, tmp_path / 'first', steps=1)
    rest = _run(config, tmp_path / 'rest', steps=3, resume=first['checkpoint'])
    a, b = state(Path(whole['checkpoint'])), state(Path(rest['checkpoint']))
    for key in ('model', 'optimizer', 'sampler', 'progress', 'rng'):
        _equal(a[key], b[key])
    assert a['config']['train']['qk_clip_source'] == 'batch_bound'
    rows = whole['history']
    assert all(row['score_bound_max'] > 0 for row in rows)
    assert [row['step'] for row in rows if 'qk_clip' in row] == [2, 4]
    assert [row['step'] for row in rest['history'] if 'qk_clip' in row] == [2, 4]
    assert all(row['qk_clip']['clipped_heads'] > 0 for row in rows if 'qk_clip' in row)
    assert [row['score_bound_max'] for row in rows[1:]] == [row['score_bound_max'] for row in rest['history']]


def test_bound_diagnostic_max_covers_all_microbatches_but_clip_uses_last(config, tmp_path, monkeypatch):
    import tnlm.qk_clip as clipping
    config.model.layers = 2
    config.train.qk_clip_source = 'batch_bound'
    config.train.qk_clip_tau = 1.
    config.train.qk_clip_every_steps = 1
    original = TensorLM.forward
    bound_values, clipped = [], []
    steps = []
    original_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        steps.append(True)
        return original_step(optimizer, *args, **kwargs)

    def forward(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        early = not bound_values
        values = (50., 999.) if early else (3., 4.)
        for layer, block in enumerate(model.blocks):
            assert block.attn.record_score_bound
            block.attn.last_score_bound = {'S1': torch.tensor([values[0]]),
                                          'S2': torch.tensor([values[1] if layer else 2.])}
        bound_values.append(max(values))

        def simulate_backward_recomputation(gradient):
            for block in model.blocks:
                block.attn.last_score_bound = {'S1': torch.tensor([10000.]), 'S2': torch.ones(1)}
            return gradient

        result[1].register_hook(simulate_backward_recomputation)
        return result

    def clip(model, tau):
        assert len(steps) == 1
        last = max(float(value.max()) for layer in score_bounds(model) for value in layer.values())
        clipped.append(last)
        return {'clipped_heads': 0, 'max_bound_before': last, 'factors': {}, 'tau': tau,
                'source': 'per-batch norm bound'}

    monkeypatch.setattr(TensorLM, 'forward', forward)
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    monkeypatch.setattr(clipping, 'qk_clip_from_bounds_', clip)
    result = _run(config, tmp_path / 'bounds')
    assert len(bound_values) > 1
    assert clipped == [4.]
    assert result['history'][0]['score_bound_max'] == 999.
    assert result['history'][0]['qk_clip']['max_bound_before'] == 4.


def test_batch_bound_tau_zero_records_without_clipping_or_probe(config, tmp_path, monkeypatch):
    config.train.qk_clip_source = 'batch_bound'
    config.train.qk_clip_tau = 0.
    config.train.qk_clip_every_steps = 0

    def forbidden(*args, **kwargs):
        raise AssertionError('diagnostic-only recording must not clip or prepare a probe')

    monkeypatch.setattr('tnlm.train.TrainingStabilityProbe', forbidden)
    monkeypatch.setattr('tnlm.qk_clip.qk_clip_from_bounds_', forbidden)
    result = _run(config, tmp_path / 'record')
    assert result['history'][0]['score_bound_max'] > 0
    assert 'qk_clip' not in result['history'][0]


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_bound_aborts_before_optimizer(config, tmp_path, monkeypatch, bad):
    config.train.qk_clip_source = 'batch_bound'
    config.train.qk_clip_tau = 0.
    original = TensorLM.forward

    def forward(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        model.blocks[0].attn.last_score_bound = {'S1': torch.tensor([bad]), 'S2': torch.ones(1)}
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError('nonfinite forward bound must abort before optimizer update')

    monkeypatch.setattr(TensorLM, 'forward', forward)
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    with pytest.raises(FloatingPointError):
        _run(config, tmp_path / 'nonfinite')
    assert not (tmp_path / 'nonfinite/final').exists()


@pytest.mark.parametrize('operation', ['sample', 'clip'])
def test_independent_probes_restore_recording_and_cached_training_bounds(config, operation):
    model = TensorLM(config.model)
    set_score_bound_recording(model, True)
    before = []
    for block in model.blocks:
        block.attn.last_score_bound = {'S1': torch.tensor([123.]), 'S2': torch.tensor([456.])}
        before.append(block.attn.last_score_bound)
    probe = TrainingStabilityProbe(config, 'cpu')
    if operation == 'sample':
        probe.sample(model)
    else:
        probe.clip(model, 1e20)
    for block, cached in zip(model.blocks, before):
        assert block.attn.record_score_bound
        assert block.attn.last_score_bound is cached
        assert float(cached['S1']) == 123. and float(cached['S2']) == 456.


@pytest.mark.parametrize('source', ['wrong', '', None, True, 0])
def test_invalid_bound_source_rejected_even_when_tau_zero(config, tmp_path, source):
    config.train.qk_clip_source = source
    config.train.qk_clip_tau = 0.
    with pytest.raises(ValueError):
        _run(config, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_bound_source_config_yaml_and_backward_default_resume_identity(config, tmp_path):
    config.train.qk_clip_source = 'probe'
    saved = full_config(config)
    saved['train'].pop('qk_clip_source', None)
    resume_validator(config)(saved)
    config.train.qk_clip_source = 'batch_bound'
    with pytest.raises(ValueError, match='resume config mismatch'):
        resume_validator(config)(saved)
    parent_config = copy.deepcopy(config)
    parent_config.train.qk_clip_source = 'probe'
    parent = _run(parent_config, tmp_path / 'parent')
    with pytest.raises(ValueError, match='resume config mismatch'):
        _run(config, tmp_path / 'changed', resume=parent['checkpoint'])
    encoded = full_config(config)
    assert encoded['train']['qk_clip_source'] == 'batch_bound'
    path = tmp_path / 'bound.yaml'
    path.write_text(yaml.safe_dump(encoded))
    restored = Config.from_yaml(path)
    assert restored.train.qk_clip_source == 'batch_bound'
