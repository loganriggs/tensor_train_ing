import copy
import json
import math

import pytest
import torch

from tnlm.optimizer_exposure import update_optimizer_exposure


def fixture():
    p = torch.nn.Parameter(torch.ones(2))
    return torch.optim.AdamW([p], lr=.1, weight_decay=.2), {'steps': 0, 'tokens': 0}


def record(progress, optimizer):
    exposure = update_optimizer_exposure(progress, optimizer)
    progress['steps'] += 1
    progress['tokens'] += 7
    return exposure


def test_analytic_shrinkage_scheduled_lrs_and_resume():
    opt, progress = fixture()
    factors = []
    for lr in [.1, .3, .025]:
        opt.param_groups[0]['lr'] = lr
        factors.append(1 - lr * .2)
        record(progress, opt)
    resumed = json.loads(json.dumps(progress))
    for state in (progress, resumed):
        opt.param_groups[0]['lr'] = .2
        exposure = record(state, opt)
        group = exposure['groups'][0]
        assert group['sum_lr'] == pytest.approx(.625)
        assert group['sum_lr_weight_decay'] == pytest.approx(.125)
        assert math.exp(group['shrinkage_log_abs']) == pytest.approx(math.prod(factors) * .96)
        assert group['shrinkage_sign'] == 1
        assert exposure['complete_history'] and exposure['observed_steps'] == 4
    assert progress == resumed
    assert not opt.state  # helper never steps or initializes optimizer state


def test_zero_negative_and_repeated_negative_factors_finite_json():
    opt, progress = fixture()
    opt.param_groups[0]['weight_decay'] = 1.
    opt.param_groups[0]['lr'] = 2.
    first = record(progress, opt)['groups'][0]
    assert first['shrinkage_sign'] == -1 and first['shrinkage_log_abs'] == 0
    assert record(progress, opt)['groups'][0]['shrinkage_sign'] == 1
    opt.param_groups[0]['lr'] = 1.
    group = record(progress, opt)['groups'][0]
    assert group['shrinkage_sign'] == 0 and group['shrinkage_log_abs'] is None
    opt.param_groups[0]['lr'] = 2.
    group = record(progress, opt)['groups'][0]
    assert group['shrinkage_sign'] == 0 and group['zero_factors'] == 1
    json.dumps(progress, allow_nan=False)


@pytest.mark.parametrize('key,value', [('lr', float('nan')), ('weight_decay', float('inf')), ('lr', -1.)])
def test_invalid_no_mutation(key, value):
    opt, progress = fixture()
    record(progress, opt)
    before = copy.deepcopy(progress)
    opt.param_groups[0][key] = value
    with pytest.raises(ValueError):
        update_optimizer_exposure(progress, opt)
    assert progress == before


def test_legacy_unknown_start_and_stale_history():
    opt, progress = fixture()
    progress.update(steps=19, tokens=135)
    exposure = record(progress, opt)
    assert not exposure['complete_history']
    assert exposure['known_start_steps'] == 19 and exposure['known_start_tokens'] == 135
    assert exposure['observed_steps'] == 1
    assert exposure['groups'][0]['sum_lr'] == .1
    progress['steps'] += 1
    before = copy.deepcopy(progress)
    with pytest.raises(ValueError, match='history'):
        update_optimizer_exposure(progress, opt)
    assert progress == before


def test_muon_raw_decay_and_group_identity():
    from tnlm.config import ModelConfig
    from tnlm.model import TensorLM
    from tnlm.optim import MuonAdamW
    model = TensorLM(ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                     d_mlp=12, vocab_size=16, padded_vocab=16, attention_backend='reference'))
    opt = MuonAdamW(model, lr=.1, aux_lr=.02, weight_decay=.2, betas=(.9, .95))
    progress = {'steps': 0, 'tokens': 0}
    exposure = record(progress, opt)
    assert exposure['groups'][0]['sum_lr_weight_decay'] == pytest.approx(.02)
    assert exposure['groups'][0]['identity']['optimizer_kind'] == 'muon'
    assert len(exposure['groups']) == len(opt.param_groups)
    before = copy.deepcopy(progress)
    opt.muon.param_groups[0]['params'].reverse()
    with pytest.raises(ValueError, match='identity'):
        update_optimizer_exposure(progress, opt)
    assert progress == before


def test_old_metrics_snapshot_and_overflow_failure_are_unchanged():
    opt, progress = fixture()
    old = record(progress, opt)
    snapshot = copy.deepcopy(old)
    record(progress, opt)
    assert old == snapshot
    before = copy.deepcopy(progress)
    opt.param_groups[0].update(lr=1e308, weight_decay=1e308)
    with pytest.raises(ValueError, match='finite'):
        update_optimizer_exposure(progress, opt)
    assert progress == before


def test_zero_lr_still_counts_successful_update():
    opt, progress = fixture()
    opt.param_groups[0]['lr'] = 0.
    group = record(progress, opt)['groups'][0]
    assert group['observed_steps'] == 1
    assert group['sum_lr'] == group['sum_lr_weight_decay'] == group['shrinkage_log_abs'] == 0.
    assert group['shrinkage_sign'] == 1
