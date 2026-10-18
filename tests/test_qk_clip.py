import pytest
import torch

from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.qk_clip import head_score_max, qk_clip_

TINY = dict(layers=2, d_model=32, heads=2, d_key=16, d_value=16, d_mlp=48, vocab_size=97, padded_vocab=128)


def _model(**kw):
    torch.manual_seed(0)
    m = TensorLM(ModelConfig(**TINY, **kw)).double().eval()
    with torch.no_grad():  # blow up head 0 of layer 1, factor S1
        m.blocks[1].attn.q1.weight[:16] *= 40
    return m


def test_clip_bounds_probe_max_and_leaves_small_heads_untouched():
    m = _model()
    idx = torch.randint(0, 97, (1, 20)); doc = torch.zeros(1, 20, dtype=torch.long)
    before = head_score_max(m, idx, doc)
    tau = 0.5 * float(before[1]["S1"][0])
    assert float(before[1]["S1"][1]) < tau  # head 1 is below the threshold
    w_head1 = m.blocks[1].attn.q1.weight[16:].clone()
    r = qk_clip_(m, idx, doc, tau)
    after = head_score_max(m, idx, doc)
    assert r["clipped_heads"] >= 1
    torch.testing.assert_close(after[1]["S1"][0], torch.tensor(tau, dtype=torch.float64), rtol=1e-6, atol=1e-9)
    assert torch.equal(m.blocks[1].attn.q1.weight[16:], w_head1)
    assert all(float(v.max()) <= tau * (1 + 1e-6) for st in after for v in st.values())


def test_clip_scales_scores_exactly_and_keeps_model_polynomial():
    m = _model(attention="lower_degree_mix")
    idx = torch.randint(0, 97, (1, 12)); doc = torch.zeros(1, 12, dtype=torch.long)
    cls = m.operator_audit()["class"]
    qk_clip_(m, idx, doc, 1e-3)  # clip everything
    assert m.operator_audit()["class"] == cls
    assert all(float(v.max()) <= 1e-3 * (1 + 1e-6) for st in head_score_max(m, idx, doc) for v in st.values())


def test_clip_rejects_spectral():
    m = _model(normalization="spectral_rms")
    with pytest.raises(NotImplementedError):
        qk_clip_(m, torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long), 1.0)


def test_score_bound_is_an_upper_bound_and_bound_clip_caps_every_score():
    from tnlm.qk_clip import qk_clip_from_bounds_, score_bounds, set_score_bound_recording
    m = _model()
    idx = torch.randint(0, 97, (3, 20)); doc = torch.zeros(3, 20, dtype=torch.long)
    set_score_bound_recording(m, True)
    m(idx, doc)
    bounds = score_bounds(m)
    for row in range(3):  # bound covers every row of the batch, not just a probe row
        true = head_score_max(m, idx[row:row + 1], doc[row:row + 1])
        for b, t in zip(bounds, true):
            for k in t:
                assert torch.all(b[k] >= t[k] * (1 - 1e-9))
    tau = 0.5 * max(float(v.max()) for st in bounds for v in st.values())
    r = qk_clip_from_bounds_(m, tau)
    assert r["clipped_heads"] >= 1
    for row in range(3):
        after = head_score_max(m, idx[row:row + 1], doc[row:row + 1])
        assert all(float(v.max()) <= tau * (1 + 1e-6) for st in after[:1] for v in st.values())  # layer 0: same inputs
    set_score_bound_recording(m, False)
    m(idx, doc)
    assert all(not st for st in score_bounds(m))
