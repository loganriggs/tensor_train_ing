"""Taper trainer lifecycle: finite policy, exact boundaries, and restored buffers."""
import copy
from types import SimpleNamespace

import pytest
import torch

from tnlm.normalization import TaperRMS
from tnlm.training_taper import TaperTraining, validate_taper_config


def config():
    return SimpleNamespace(model=SimpleNamespace(normalization='taper', final_normalization='taper',
        taper_ema_rate=.1, activation_checkpointing=False),
        train=SimpleNamespace(taper_anchor_lambda=.1, taper_end_fraction=1., compile=False))


def model():
    return torch.nn.Sequential(TaperRMS(4, ema_rate=.1), TaperRMS(4, ema_rate=.1)).double()


@pytest.mark.parametrize('owner,field,value', [
    ('train', 'taper_anchor_lambda', -1), ('train', 'taper_anchor_lambda', True),
    ('train', 'taper_anchor_lambda', float('nan')), ('train', 'taper_end_fraction', 0),
    ('train', 'taper_end_fraction', 1.1), ('train', 'taper_end_fraction', float('inf')),
    ('model', 'taper_ema_rate', 0), ('model', 'taper_ema_rate', 1.1),
    ('model', 'taper_ema_rate', False), ('train', 'compile', True),
    ('model', 'activation_checkpointing', True)])
def test_policy_rejects_invalid(owner, field, value):
    cfg = config()
    setattr(getattr(cfg, owner), field, value)
    with pytest.raises(ValueError):
        validate_taper_config(cfg)


@pytest.mark.parametrize('total,warmup,fraction', [(100,100,1.), (101,100,.5), (110,0,1.), (110,True,1.)])
def test_endpoint_requires_post_warmup_steps(total, warmup, fraction):
    cfg = config()
    cfg.train.taper_end_fraction = fraction
    with pytest.raises(ValueError):
        TaperTraining(model(), cfg, total_steps=total, warmup_steps=warmup)


def update(m, helper, k):
    helper.before_step(k)
    helper.set_valid_mask(torch.tensor([[True, False]]))
    x = torch.tensor([[[1.,2.,3.,4.], [100.,200.,300.,400.]]], dtype=torch.float64, requires_grad=True)
    m(x)
    anchor = helper.anchor()
    assert helper.anchor() is None
    helper.set_valid_mask(None)
    assert all(not norm.collecting and norm._valid is None for norm in helper.norms)
    helper.after_step(k + 1)
    return helper.metrics(0. if anchor is None else anchor.detach())


def test_boundaries_masks_and_final_gate():
    m = model()
    helper = TaperTraining(m, config(), total_steps=4, warmup_steps=2)
    a = update(m, helper, 0)
    assert a['taper_gate'] == 1 and 'taper_constants' not in a
    assert all(float(norm.ema_steps) == 1 and not bool(norm.frozen) for norm in m)
    b = update(m, helper, 1)
    assert b['taper_gate'] == 1 and set(b['taper_constants']) == {'0', '1'}
    assert float(m[0].s_tgt) == pytest.approx((7.5 + m[0].eps)**.5)
    c = update(m, helper, 2)
    assert c['taper_gate_used'] == 1 and c['taper_gate'] == pytest.approx(.5)
    assert 'taper_constants' not in c
    d = update(m, helper, 3)
    assert d['taper_gate_used'] == pytest.approx(.5) and d['taper_gate'] == 0
    assert all(float(norm.ema_steps) == 2 and norm.op_class == 'P' for norm in m)
    with pytest.raises(ValueError, match='exactly one'):
        helper.after_step(4)


@pytest.mark.parametrize('split', [1,2,3])
def test_restored_state_and_boundary_freeze_match(split):
    whole = model()
    h = TaperTraining(whole, config(), total_steps=4, warmup_steps=2)
    saved = None
    for k in range(4):
        update(whole, h, k)
        if k + 1 == split:
            saved = copy.deepcopy(whole.state_dict())
    resumed = model()
    resumed.load_state_dict(saved)
    r = TaperTraining(resumed, config(), total_steps=4, warmup_steps=2)
    for k in range(split, 4):
        update(resumed, r, k)
    for key, value in whole.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key


def test_all_commits_precede_finite_failure(monkeypatch):
    m = model()
    h = TaperTraining(m, config(), total_steps=4, warmup_steps=2, distributed=True)
    calls = []
    for i, norm in enumerate(m):
        def commit(distributed, norm=norm, i=i):
            calls.append((i, distributed))
            norm.ema_num.fill_(float('nan'))
        monkeypatch.setattr(norm, 'commit', commit)
    h.before_step(0)
    with pytest.raises(FloatingPointError):
        h.after_step(1)
    assert calls == [(0, True), (1, True)]


def test_zero_target_anchor_drains_without_loss():
    m = model()
    h = TaperTraining(m, config(), total_steps=3, warmup_steps=1)
    update(m, h, 0)
    h.before_step(1)
    h.set_valid_mask(torch.zeros(1,2,dtype=torch.bool))
    m(torch.randn(1,2,4,dtype=torch.float64))
    assert h.anchor() is None
    assert all(n._anchor_sum is None and n._anchor_count == 0 for n in m)


def test_disabled_policy_has_no_effect():
    cfg = config()
    cfg.model.normalization, cfg.model.final_normalization = 'pre_rms', 'rms'
    cfg.train.compile = True
    h = TaperTraining(torch.nn.Linear(4,4), cfg, total_steps=1, warmup_steps=100)
    h.before_step(0)
    h.set_valid_mask(None)
    h.after_step(1)
    assert not h.enabled and h.anchor() is None and h.metrics(0.) == {}
