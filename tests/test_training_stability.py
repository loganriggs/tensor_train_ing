"""Fixed-row stability logging preserves the training lifecycle and RNG."""
import json
import random

import numpy as np
import pytest
import torch

from tnlm.data import PackedSampler, batch_hash
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules
from tnlm.training_stability import TrainingStabilityProbe, stability_interval


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng(expected):
    actual = _rng()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


def test_interval_defaults_off_and_strict(config):
    assert stability_interval(config) == 0
    for value in (0, 1, 20):
        config.train.stability_every_steps = value
        assert stability_interval(config) == value
    for value in (True, False, -1, 1.0, '1', None):
        config.train.stability_every_steps = value
        with pytest.raises(ValueError, match='stability_every_steps'):
            stability_interval(config)


@pytest.mark.parametrize('normalization', ['pre_rms', 'lagged'])
def test_real_stats_preserve_model_modes_rng_and_fixed_training_row(config, normalization):
    config.model.normalization = normalization
    if normalization == 'lagged':
        config.model.final_normalization = 'lagged'
    model = TensorLM(config.model)
    model.train()
    model.blocks[0].mlp.eval()  # Mixed flags must not be flattened on restoration.
    norms = list(lagged_modules(model))
    for norm in norms:
        norm.collecting = True
        norm.acc_count.fill_(3)
        norm.acc_sum.fill_(7)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = _rng()
    sampler = PackedSampler(config.raw['data']['manifest'], context=4, seed=config.train.seed)
    expected_hash = batch_hash(sampler.next_batch())
    probe = TrainingStabilityProbe(config, 'cpu')
    first = probe.sample(model)
    second = probe.sample(model)
    _assert_rng(rng)
    assert first['stats'] == second['stats']
    assert first['probe']['batch_sha256'] == expected_hash
    assert first['probe']['split'] == 'train'
    assert first['probe']['seed'] == config.train.seed
    assert len(first['probe']['manifest_sha256']) == 64
    assert first['stats']['tokens'] == 4
    assert len(first['stats']['per_layer']) == config.model.layers
    assert first['seconds'] > 0
    assert first['nonfinite_paths'] == []
    json.dumps(first, allow_nan=False)
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    for norm in norms:
        assert norm.collecting and norm.acc_count == 3 and norm.acc_sum == 7
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_context_cap_and_independent_training_sampler(config):
    config.raw['data']['context'] = 1024
    sampler = PackedSampler(config.raw['data']['manifest'], context=4, seed=config.train.seed)
    expected = PackedSampler(config.raw['data']['manifest'], context=4, seed=config.train.seed)
    sampler.next_batch()
    expected.next_batch()
    probe = TrainingStabilityProbe(config, 'cpu')
    assert probe.identity['context'] == 512
    assert probe.idx.shape == (1, 512)
    assert batch_hash(sampler.next_batch()) == batch_hash(expected.next_batch())


@pytest.mark.parametrize('fail', [False, True])
def test_diagnostic_failure_or_nonfinite_restores_lifecycle(config, monkeypatch, fail):
    config.model.normalization = 'lagged'
    model = TensorLM(config.model)
    model.train()
    model.blocks[0].mlp.eval()
    modes = [module.training for module in model.modules()]
    norms = list(lagged_modules(model))
    valid = torch.ones((1, 4), dtype=torch.bool)
    for norm in norms:
        norm.collecting = True
        norm._valid = valid
    probe = TrainingStabilityProbe(config, 'cpu')
    rng = _rng()

    def diagnostic(model, idx, doc):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert all(not module.collecting for module in norms)
        random.random()
        np.random.random()
        torch.rand(7)
        for norm in norms:
            norm._valid = None
        if fail:
            raise RuntimeError('probe failure')
        return {'per_layer': [{'score': float('nan')}], 'worst': float('inf'), 'negative': -float('inf')}

    monkeypatch.setattr('tnlm.analysis.stability_stats', diagnostic)
    if fail:
        with pytest.raises(RuntimeError, match='probe failure'):
            probe.sample(model)
    else:
        result = probe.sample(model)
        assert result['stats'] == {'per_layer': [{'score': None}], 'worst': None, 'negative': None}
        assert result['nonfinite_paths'] == ['stats.per_layer[0].score', 'stats.worst', 'stats.negative']
        json.dumps(result, allow_nan=False)
    _assert_rng(rng)
    assert modes == [module.training for module in model.modules()]
    assert all(norm.collecting and norm._valid is valid for norm in norms)


@pytest.mark.parametrize('failure', ['exception', 'nonfinite'])
def test_clip_restores_probe_lifecycle_and_rng_on_failure(config, monkeypatch, failure):
    config.model.normalization = config.model.final_normalization = 'lagged'
    model = TensorLM(config.model)
    model.train();model.blocks[0].mlp.eval()
    norms = list(lagged_modules(model))
    for norm in norms:
        norm.collecting = True
    modes = [module.training for module in model.modules()]
    probe = TrainingStabilityProbe(config, 'cpu')
    rng = _rng()
    def clip(model, idx, doc, tau):
        assert all(not module.training for module in model.modules())
        assert all(not norm.collecting for norm in norms)
        random.random();np.random.random();torch.rand(3)
        if failure == 'exception':raise RuntimeError('clipping failure')
        return {'max_before': float('inf')}
    monkeypatch.setattr('tnlm.qk_clip.qk_clip_', clip)
    with pytest.raises(RuntimeError if failure == 'exception' else FloatingPointError):
        probe.clip(model, 1.)
    _assert_rng(rng)
    assert modes == [module.training for module in model.modules()]
    assert all(norm.collecting for norm in norms)
