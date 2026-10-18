"""Real CPU integration for warmup calibration, taper anchors, and replay."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import state
from tnlm.model import TensorLM
from tnlm.normalization import taper_modules
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


def _configure(config):
    config.model.normalization = config.model.final_normalization = 'taper'
    config.train.compile = False
    config.train.lr = .0001
    config.train.rung_total_tokens = 713  # 101 complete seven-target updates + six.
    config.train.taper_anchor_lambda = .1
    config.train.taper_end_fraction = 1.
    config.raw['project'].update(keep_recovery_checkpoints=0, milestone_tokens=[700, 707])
    return config


def _run(config, path, steps=102, **kwargs):
    return train_run(config, path, max_steps=steps, max_seconds=120, device='cpu', **kwargs)


def test_taper_real_freeze_gate_masks_and_exact_warmup_boundary_replay(config, tmp_path, monkeypatch):
    _configure(config)
    masks_seen = {'warmup': 0, 'frozen': 0}
    ce_forwards = []
    original = TensorLM.forward

    def inspected_forward(model, idx, doc_ids=None, targets=None, **kwargs):
        if model.training and targets is not None:
            for norm in taper_modules(model):
                frozen = bool(norm.frozen)
                expected = targets != -100 if frozen else doc_ids >= 0
                assert norm._valid is not None
                assert torch.equal(norm._valid, expected)
                assert norm.collecting is not frozen
                masks_seen['frozen' if frozen else 'warmup'] += 1
        result = original(model, idx, doc_ids=doc_ids, targets=targets, **kwargs)
        if model.training and targets is not None:
            ce_forwards.append((float(result[1].detach()), int(result[2])))
        return result

    monkeypatch.setattr(TensorLM, 'forward', inspected_forward)
    whole = _run(config, tmp_path / 'whole')
    prefix = _run(config, tmp_path / 'prefix', steps=99)
    resumed = _run(config, tmp_path / 'resumed', steps=3, resume=prefix['checkpoint'])
    expected, actual = state(Path(whole['checkpoint'])), state(Path(resumed['checkpoint']))
    for key in ('model', 'optimizer', 'sampler', 'progress', 'rng'):
        _equal(actual[key], expected[key])
    assert whole['progress']['steps'] == 102 and whole['progress']['tokens'] == 713
    assert whole['history'][-1]['prediction_tokens'] == 6
    assert any(row['microbatches'] > 1 for row in whole['history'])
    assert all(masks_seen.values())
    for filename, step, gate, frozen in [('prefix/final', 99, 1., False),
            ('whole/milestone-000000000700', 100, 1., True),
            ('whole/milestone-000000000707', 101, .5, True), ('whole/final', 102, 0., True)]:
        payload = state(tmp_path / filename)
        model = TensorLM(config.model)
        model.load_state_dict(payload['model'])
        for norm in taper_modules(model):
            assert float(norm.ema_steps) == min(step, 100)
            assert bool(norm.frozen) is frozen
            assert float(norm.gate) == pytest.approx(gate, abs=1e-15)
        assert payload['extra']['operator_audit']['class'] == ('P' if gate == 0 else 'N')
    assert whole['operator_audit']['class'] == 'P'
    rows = whole['history']
    cursor = 0
    for row in rows:
        micros = ce_forwards[cursor:cursor + row['microbatches']]
        cursor += row['microbatches']
        assert sum(count for _, count in micros) == row['prediction_tokens']
        assert row['loss'] == pytest.approx(sum(loss for loss, _ in micros) / row['prediction_tokens'])
    assert all(row['taper_anchor'] == 0. for row in rows[:100])
    assert all(row['taper_anchor'] > 0. for row in rows[100:])
    assert rows[99]['taper_gate'] == 1. and rows[100]['taper_gate'] == pytest.approx(.5)
    assert rows[-1]['taper_gate'] == 0.
    assert rows[100]['taper_gate_used'] == 1. and rows[101]['taper_gate_used'] == pytest.approx(.5)
    constants = [row['taper_constants'] for row in rows if 'taper_constants' in row]
    assert len(constants) == 1 and constants[0]
    # Cooldown resets the LR scheduler only: Taper uses cumulative update count.
    cooldown_parent = tmp_path / 'whole/milestone-000000000700'
    cooldown = _run(config, tmp_path / 'cooldown', steps=2, resume=cooldown_parent,
                    cooldown_steps=2, cooldown_end_tokens=713)
    cool_first = _run(config, tmp_path / 'cool-first', steps=1, resume=cooldown_parent,
                      cooldown_steps=2, cooldown_end_tokens=713)
    cool_resumed = _run(config, tmp_path / 'cool-resumed', steps=1, resume=cool_first['checkpoint'],
                        cooldown_steps=2, cooldown_end_tokens=713)
    assert cooldown['progress']['steps'] == 102
    assert cooldown['progress']['scheduler_step'] == 2
    assert [row['taper_gate_used'] for row in cooldown['history']] == pytest.approx([1., .5])
    assert [row['taper_gate'] for row in cooldown['history']] == pytest.approx([.5, 0.])
    assert cooldown['operator_audit']['class'] == 'P'
    cool_expected, cool_actual = state(Path(cooldown['checkpoint'])), state(Path(cool_resumed['checkpoint']))
    for key in ('model', 'optimizer', 'sampler', 'progress', 'rng'):
        _equal(cool_actual[key], cool_expected[key])
    promoted = copy.deepcopy(config)
    promoted.train.rung_total_tokens += 7
    with pytest.raises(ValueError):
        _run(promoted, tmp_path / 'promotion', steps=1, resume=prefix['checkpoint'])
    with pytest.raises(ValueError):
        _run(config, tmp_path / 'batch-branch', steps=1,
             resume=tmp_path / 'whole/milestone-000000000700', batch_tokens=14, lr_scale=1.)


@pytest.mark.parametrize('change', ['short_horizon', 'compile', 'activation_checkpointing',
                                   'negative_anchor', 'nan_anchor', 'bad_end_fraction'])
def test_invalid_taper_training_rejected_before_output(config, tmp_path, change):
    _configure(config)
    if change == 'short_horizon':
        config.train.rung_total_tokens = 700
    elif change == 'compile':
        config.train.compile = True
    elif change == 'activation_checkpointing':
        config.model.activation_checkpointing = True
    elif change == 'negative_anchor':
        config.train.taper_anchor_lambda = -.1
    elif change == 'nan_anchor':
        config.train.taper_anchor_lambda = float('nan')
    else:
        config.train.taper_end_fraction = .5  # Taper end precedes warmup end.
    with pytest.raises((ValueError, NotImplementedError)):
        _run(config, tmp_path / 'bad', steps=1)
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('operation', ['sample', 'clip'])
def test_taper_probe_does_not_collect_warmup_statistics(config, operation):
    _configure(config)
    model = TensorLM(config.model)
    model.train()
    norms = taper_modules(model)
    valid = torch.ones((1, 4), dtype=torch.bool)
    for norm in norms:
        norm.collecting = True
        norm.set_valid_mask(valid)
        norm.acc_count.fill_(7)
        norm.acc_r.fill_(3)
    before = [(norm.acc_num.clone(), norm.acc_den.clone(), norm.acc_r.clone(), norm.acc_count.clone())
              for norm in norms]
    probe = TrainingStabilityProbe(config, 'cpu')
    if operation == 'sample':
        probe.sample(model)
    else:
        probe.clip(model, 1e20)  # Exercise the wrapper without changing weights.
    for norm, saved in zip(norms, before):
        assert norm.collecting and norm._valid is valid and norm.training
        _equal((norm.acc_num, norm.acc_den, norm.acc_r, norm.acc_count), saved)
        assert norm._anchor_sum is None and norm._anchor_count == 0.
