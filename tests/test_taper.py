import math

import pytest
import torch

from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.normalization import TaperRMS, taper_anchor_loss, taper_gate, taper_modules, taper_set_gate_

TINY = dict(layers=2, d_model=32, heads=2, d_key=16, d_value=16, d_mlp=48, vocab_size=97, padded_vocab=128)


def _calibrate(m, idx, doc, steps=5):
    for t in taper_modules(m):
        t.collecting = True
    for _ in range(steps):
        m(idx, doc)
        for t in taper_modules(m):
            t.commit(distributed=False)
    for t in taper_modules(m):
        t.collecting = False
        t.freeze()


def test_gate_one_is_rmsnorm_and_gate_zero_is_fixed_diagonal():
    torch.manual_seed(0)
    n = TaperRMS(8).double()
    with torch.no_grad():
        n.weight.uniform_(0.5, 1.5)
    x = torch.randn(3, 5, 8, dtype=torch.float64)
    r = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + n.eps)
    torch.testing.assert_close(n(x), x / r * n.weight)
    n.collecting = True; n(x); n.commit(distributed=False); n.collecting = False; n.freeze()
    n.eval(); n.gate.fill_(0.0)
    y = n(x)
    torch.testing.assert_close(y, float(n.c) * x * n.weight_tilde)                 # sample-independent
    torch.testing.assert_close(n(3.0 * x), 3.0 * y)                                  # exactly linear (1-homogeneous)
    assert n.op_class == "P"


def test_calibration_is_least_squares_match_and_bias_corrected():
    torch.manual_seed(1)
    n = TaperRMS(16, ema_rate=0.3).double()
    xs = [torch.randn(4, 7, 16, dtype=torch.float64) * s for s in (0.5, 1.0, 2.0)]
    n.collecting = True
    for x in xs:
        n(x); n.commit(distributed=False)
    n.freeze()
    # with exact bias correction a constant statistic is recovered exactly; check s_tgt against the EMA definition
    mu, w = 0.3, [(1 - 0.3) ** (2 - i) * 0.3 for i in range(3)]
    rbar = [torch.sqrt(x.pow(2).mean(-1) + n.eps).mean() for x in xs]
    expect = sum(wi * ri for wi, ri in zip(w, rbar)) / (1 - (1 - mu) ** 3)
    torch.testing.assert_close(n.s_tgt, expect)
    assert 0 < float(n.c) and math.isfinite(float(n.c))


def test_all_taper_model_is_polynomial_once_frozen_and_anchor_loss_is_differentiable():
    torch.manual_seed(0)
    m = TensorLM(ModelConfig(**TINY, normalization="taper", final_normalization="taper")).double()
    idx = torch.randint(0, 97, (2, 12)); doc = torch.zeros(2, 12, dtype=torch.long)
    assert m.operator_audit()["class"] == "N"
    _calibrate(m, idx, doc)
    taper_set_gate_(m, 0.5)
    m.train()
    m(idx, doc)
    a = taper_anchor_loss(m)
    assert a is not None and a.requires_grad and torch.isfinite(a)
    a.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert taper_anchor_loss(m) is None                                               # accumulators reset
    taper_set_gate_(m, 0.0)
    m.eval()
    assert m.operator_audit()["class"] == "P"
    assert all(t.op_class == "P" for t in taper_modules(m)) and len(taper_modules(m)) == 2 * 2 + 1


def test_taper_requires_tapered_final_norm_and_no_activation_checkpointing():
    with pytest.raises(ValueError):
        ModelConfig(**TINY, normalization="taper")
    with pytest.raises(ValueError):
        ModelConfig(**TINY, normalization="taper", final_normalization="taper", activation_checkpointing=True)


def test_gate_schedule():
    assert taper_gate(0, 10, 110) == 1.0 and taper_gate(10, 10, 110) == 1.0
    assert taper_gate(60, 10, 110) == pytest.approx(0.5) and taper_gate(110, 10, 110) == 0.0 and taper_gate(500, 10, 110) == 0.0


def test_freeze_without_statistics_raises():
    with pytest.raises(RuntimeError):
        TaperRMS(4).freeze()
