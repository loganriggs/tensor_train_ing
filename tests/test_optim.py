import copy

import torch

from tnlm.config import ModelConfig, TrainConfig
from tnlm.model import TensorLM
from tnlm.optim import MuonAdamW, make_optimizer, param_split

from helpers import TINY_MODEL as TINY


def _model(**kw):
    torch.manual_seed(0)
    return TensorLM(ModelConfig(**TINY, **kw))


def test_split_keeps_logical_matrices_separate():
    m = _model(attention="lower_degree_mix", mlp="affine_bilinear")
    hidden, vocab, small = param_split(m)
    names = {id(p): n for n, p in m.named_parameters()}
    hn = {names[id(p)] for p in hidden}
    assert "blocks.0.mlp.l.weight" in hn and "blocks.0.mlp.r.weight" in hn and "blocks.0.attn.q2.weight" in hn
    assert all(p.ndim == 2 for p in hidden)
    assert {names[id(p)] for p in vocab} == {"embed.weight", "unembed.weight"}
    assert any("coef" in names[id(p)] for p in small) and any("bias" in names[id(p)] for p in small)
    assert len(hidden) + len(vocab) + len(small) == len(list(m.parameters()))


def test_muon_trains_and_state_roundtrips():
    m = _model()
    opt = make_optimizer(m, TrainConfig(optimizer="muon", lr=0.02, aux_adamw_lr=3e-3))
    assert isinstance(opt, MuonAdamW) and {g["kind"] for g in opt.param_groups} == {"muon", "adamw_vocab", "adamw_small"}
    idx = torch.randint(0, 97, (4, 16))
    losses = []
    for i in range(12):
        if i == 6:
            snap_m, snap_o = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
        _, loss = m(idx, targets=idx.roll(-1, 1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    final = copy.deepcopy(m.state_dict())
    # resume from step 6 reproduces steps 6..11 exactly
    m2 = _model(); m2.load_state_dict(snap_m)
    opt2 = make_optimizer(m2, TrainConfig(optimizer="muon", lr=0.02, aux_adamw_lr=3e-3))
    opt2.load_state_dict(snap_o)
    for _ in range(6):
        _, loss = m2(idx, targets=idx.roll(-1, 1))
        opt2.zero_grad(); loss.backward(); opt2.step()
    for k in final:
        torch.testing.assert_close(m2.state_dict()[k], final[k], rtol=0, atol=0)


def test_adamw_groups_decay_only_hidden_matrices():
    m = _model()
    opt = make_optimizer(m, TrainConfig())
    dec = [g for g in opt.param_groups if g["weight_decay"] > 0]
    assert len(dec) == 1 and all(p.ndim == 2 for p in dec[0]["params"])
    assert all(p is not m.embed.weight and p is not m.unembed.weight for p in dec[0]["params"])


def test_width_scaling_lr_and_logit_multiplier():
    m = _model(width_scaling_base=8)  # d_model 32 -> r = 1/4
    assert abs(m.logit_multiplier - 0.25 ** 0.5) < 1e-12
    opt = make_optimizer(m, TrainConfig(lr=1e-3))
    hid = [g for g in opt.param_groups if g["weight_decay"] > 0][0]
    assert abs(hid["base_lr"] - 1e-3 * 0.5) < 1e-15 and abs(hid["lr"] - 5e-4) < 1e-15
    assert abs(hid["weight_decay"] * hid["lr"] - 0.1 * 1e-3) < 1e-12  # decoupled decay per step unchanged
    other = [g for g in opt.param_groups if g["weight_decay"] == 0][0]
    assert other["base_lr"] == 1e-3
    # chunked-CE path applies the same multiplier as the dense path
    idx = torch.randint(0, 97, (2, 9))
    dense = _model(width_scaling_base=8)
    chunk = _model(width_scaling_base=8, loss_chunk_tokens=5)
    chunk.load_state_dict(dense.state_dict())
    torch.testing.assert_close(chunk(idx, targets=idx, reduction="sum")[1], dense(idx, targets=idx, reduction="sum")[1])


def test_depth_scaling_composes_with_width_scaling():
    m = _model(width_scaling_base=8, depth_scaling_base=8)  # d 32 -> (1/4)^.5 ; layers 2 -> (8/2)^.5
    opt = make_optimizer(m, TrainConfig(lr=1e-3))
    hid = [g for g in opt.param_groups if g["weight_decay"] > 0][0]
    assert abs(hid["lr"] - 1e-3 * 0.5 * 2.0) < 1e-15


def test_coordcheck_train_kw_selects_muon_spectral_scaling():
    from tnlm.coordcheck import run
    kw = dict(widths=(64, 128), layers=1, steps=1, batch=2, context=16, vocab=256, device="cpu", optimizer="muon")
    a = run(lr=1e-3, **kw)
    b = run(lr=1e-3, train_kw={"muon_adjust_lr": "spectral_unclamped"}, **kw)
    # same seed and data; only the Muon LR rule differs, so the hidden updates must differ
    assert a["rows"][128]["resid0"]["delta"] != b["rows"][128]["resid0"]["delta"]
    assert a["loglog_slopes"]["resid0.delta"] > b["loglog_slopes"]["resid0.delta"]
