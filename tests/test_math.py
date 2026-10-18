"""check_math: CPU FP64 algebra, Jacobian, gradient and causality tests (brief §3, §8, §10, §11)."""
import itertools
import math

import pytest
import torch

from tnlm.attention import (polynomial_attention, polynomial_attention_recurrent, row_scale_mask,
                            scores, visible_mask)
from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.normalization import LaggedRMS, RMSNorm, SpectralLinear, freeze_spectral, lagged_modules, rms_jacobian

torch.manual_seed(0)
D64 = torch.float64


def r0(x):  # gain-free, zero-eps RMS
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


def bilinear(L, R, Dm):
    return lambda x: ((x @ L.T) * (x @ R.T)) @ Dm.T


# ---------------------------------------------------------------- normalization
def test_rms_jacobian_matches_autograd():
    d = 16
    x, g = torch.randn(d, dtype=D64), torch.rand(d, dtype=D64) + 0.5
    norm = RMSNorm(d, eps=1e-3).double()
    norm.weight.data = g
    J = torch.autograd.functional.jacobian(norm, x)
    torch.testing.assert_close(J, rms_jacobian(x, g, 1e-3), rtol=1e-10, atol=1e-12)


def test_metric_norm_jacobian_and_nullspace():
    d, k, eps = 12, 3, 1e-4
    P = torch.randn(k, d, dtype=D64)
    M = P.T @ P / k
    g = torch.rand(d, dtype=D64) + 0.5
    f = lambda x: g * x / torch.sqrt(x @ M @ x + eps)
    x = torch.randn(d, dtype=D64)
    r = torch.sqrt(x @ M @ x + eps)
    J = torch.diag(g) / r - torch.outer(g * x, M @ x) / r ** 3
    torch.testing.assert_close(torch.autograd.functional.jacobian(f, x), J, rtol=1e-10, atol=1e-12)
    # M delta = 0 => denominator exactly unchanged along x + t delta
    delta = torch.linalg.svd(P, full_matrices=True).Vh[-1]
    for t in (0.1, 3.0, -50.0):
        y = x + t * delta
        torch.testing.assert_close(y @ M @ y, x @ M @ x, rtol=1e-10, atol=1e-10)


def test_group_rms_is_block_diagonal_dependency():
    from tnlm.normalization import GroupRMSNorm
    d, G = 16, 4
    n = GroupRMSNorm(d, G).double()
    x = torch.randn(d, dtype=D64)
    J = torch.autograd.functional.jacobian(n, x)
    blk = torch.block_diag(*[torch.ones(d // G, d // G, dtype=D64)] * G)
    assert J[blk == 0].abs().max() == 0


# ---------------------------------------------------------------- bilinear structure
def test_bilinear_local_tensor_and_symmetrization():
    d, h, o = 6, 5, 4
    L, R, Dm = (torch.randn(h, d, dtype=D64), torch.randn(h, d, dtype=D64), torch.randn(o, h, dtype=D64))
    T = torch.einsum("oh,ha,hb->oab", Dm, L, R)
    Ts = 0.5 * (T + T.transpose(1, 2))
    x = torch.randn(3, d, dtype=D64)
    y = bilinear(L, R, Dm)(x)
    torch.testing.assert_close(torch.einsum("oab,na,nb->no", T, x, x), y)
    torch.testing.assert_close(torch.einsum("oab,na,nb->no", Ts, x, x), y)
    # antisymmetric part does not affect the function
    Ta = T - Ts
    assert torch.einsum("oab,na,nb->no", Ta, x, x).abs().max() < 1e-12


def test_affine_bilinear_via_homogeneous_coordinate():
    d, h, o = 5, 4, 3
    L, R, Dm = (torch.randn(h, d, dtype=D64), torch.randn(h, d, dtype=D64), torch.randn(o, h, dtype=D64))
    bL, bR, bD = torch.randn(h, dtype=D64), torch.randn(h, dtype=D64), torch.randn(o, dtype=D64)
    x = torch.randn(d, dtype=D64)
    y = Dm @ ((L @ x + bL) * (R @ x + bR)) + bD
    # append constant coordinate 1; fold bD into a channel with l=r=e_const
    e = torch.zeros(1, d + 1, dtype=D64); e[0, d] = 1
    La = torch.cat([torch.cat([L, bL[:, None]], 1), e])
    Ra = torch.cat([torch.cat([R, bR[:, None]], 1), e])
    Da = torch.cat([Dm, bD[:, None]], 1)
    xa = torch.cat([x, torch.ones(1, dtype=D64)])
    T = torch.einsum("oh,ha,hb->oab", Da, La, Ra)
    torch.testing.assert_close(torch.einsum("oab,a,b->o", T, xa, xa), y)


def test_homogeneous_cancellation_and_counterexamples():
    d = 8
    F = bilinear(torch.randn(10, d, dtype=D64), torch.randn(10, d, dtype=D64), torch.randn(d, 10, dtype=D64))
    x = torch.randn(4, d, dtype=D64) * 3
    torch.testing.assert_close(r0(F(r0(x))), r0(F(x)))  # holds for a single homogeneous branch
    # residual breaks it
    assert (r0(x + F(r0(x))) - r0(x + F(x))).abs().max() > 1e-3
    # finite eps breaks it
    reps = lambda z: z / (z.pow(2).mean(-1, keepdim=True) + 1.0).sqrt()
    assert (reps(F(reps(x))) - reps(F(x))).abs().max() > 1e-6


# ---------------------------------------------------------------- attention
def _qkv(B=2, H=3, T=7, dk=4, dv=5, dtype=D64, req=True):
    t = [torch.randn(B, H, T, dk, dtype=dtype, requires_grad=req) for _ in range(4)]
    v = torch.randn(B, H, T, dv, dtype=dtype, requires_grad=req)
    return (*t, v)


@pytest.mark.parametrize("kind", ["signed_product", "offset_square", "lower_degree_mix"])
def test_attention_gradcheck(kind):
    q1, k1, q2, k2, v = _qkv()
    doc = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]])
    C = row_scale_mask(visible_mask(doc), 1.0 if kind == "offset_square" else 0.5, D64)
    H = q1.shape[1]
    coef = torch.randn(H, 4, dtype=D64, requires_grad=True)
    b = torch.randn(H, dtype=D64, requires_grad=True)
    f = lambda *a: polynomial_attention(kind, *a[:5], C, coef=a[5], b=a[6])
    assert torch.autograd.gradcheck(f, (q1, k1, q2, k2, v, coef, b))


def test_attention_analytic_backward():
    """Brief §8 formulas: grad_V = P^T U, grad_S1 = H*S2, grad_S2 = H*S1, H = C*(U V^T)."""
    q1, k1, q2, k2, v = _qkv()
    doc = torch.zeros(2, 7, dtype=torch.long)
    C = row_scale_mask(visible_mask(doc), 0.5, D64)
    S1, S2 = scores(q1, k1), scores(q2, k2)
    S1.retain_grad(); S2.retain_grad()
    y = (C * S1 * S2) @ v
    U = torch.randn_like(y)
    y.backward(U)
    P = C * S1 * S2
    Hm = C * (U @ v.transpose(-1, -2))
    torch.testing.assert_close(v.grad, P.transpose(-1, -2) @ U)
    torch.testing.assert_close(S1.grad, Hm * S2)
    torch.testing.assert_close(S2.grad, Hm * S1)


def test_recurrent_equals_direct():
    q1, k1, q2, k2, v = [t.detach()[0] for t in _qkv(B=1, T=9)]
    doc = torch.zeros(1, 9, dtype=torch.long)
    C = row_scale_mask(visible_mask(doc), 0.5, D64)
    direct = polynomial_attention("signed_product", q1[None], k1[None], q2[None], k2[None], v[None], C)[0]
    scale = torch.arange(1, 10, dtype=D64).pow(-0.5)
    rec = polynomial_attention_recurrent(q1, k1, q2, k2, v, scale)
    torch.testing.assert_close(rec, direct, rtol=1e-12, atol=1e-12)


def test_masked_entries_zero_after_polynomial():
    """Offset/constant terms must not leak through masked (future / other-doc) entries."""
    q1, k1, q2, k2, v = [t.detach() for t in _qkv(B=1, T=6)]
    doc = torch.tensor([[0, 0, 1, 1, 1, 1]])
    C = row_scale_mask(visible_mask(doc), 1.0, D64)
    b = torch.full((q1.shape[1],), 5.0, dtype=D64)
    y = polynomial_attention("offset_square", q1, k1, q2, k2, v, C, b=b)
    v2 = v.clone(); v2[:, :, 4:] += 100  # future positions
    v2[:, :, :2] += 100  # other document
    y2 = polynomial_attention("offset_square", q1, k1, q2, k2, v2, C, b=b)
    torch.testing.assert_close(y[:, :, 2:4], y2[:, :, 2:4])


def test_pre_rms_attention_factor():
    """Pre-RMS (gain-free) signed attention: contribution j->i = raw contribution / (r_i^2 r_j^3)."""
    d, dk, dv, T = 10, 4, 3, 5
    W = [torch.randn(dk, d, dtype=D64) for _ in range(4)]
    Wv = torch.randn(dv, d, dtype=D64)
    x = torch.randn(T, d, dtype=D64) * torch.rand(T, 1, dtype=D64).mul(5).add(0.2)
    r = x.pow(2).mean(-1).sqrt()

    def contrib(z, i, j):
        s1 = (W[0] @ z[i]) @ (W[1] @ z[j])
        s2 = (W[2] @ z[i]) @ (W[3] @ z[j])
        return s1 * s2 * (Wv @ z[j])

    xn = x / r[:, None]
    for i, j in [(4, 1), (3, 3), (2, 0)]:
        torch.testing.assert_close(contrib(xn, i, j), contrib(x, i, j) / (r[i] ** 2 * r[j] ** 3))


# ---------------------------------------------------------------- full model
from helpers import TINY_MODEL as TINY
COMBOS = [("signed_product", "bilinear", p) for p in
          ("pre_rms", "branch_rms", "sandwich_rms", "group_rms", "spectral_rms", "lagged")] + [
    ("offset_square", "bilinear", "pre_rms"), ("lower_degree_mix", "affine_bilinear", "pre_rms"),
    ("softmax", "swiglu", "pre_rms")]


def tiny(attn, mlp, pkg, **kw):
    fin = "lagged" if pkg == "lagged" else "rms"
    cfg = ModelConfig(**TINY, attention=attn, mlp=mlp, normalization=pkg, final_normalization=fin, **kw)
    torch.manual_seed(1)
    return TensorLM(cfg).double().eval()


@pytest.mark.parametrize("attn,mlp,pkg", COMBOS)
def test_causal_prefix_invariance(attn, mlp, pkg):
    m = tiny(attn, mlp, pkg)
    idx = torch.randint(0, 97, (2, 12))
    idx2 = idx.clone(); idx2[:, 7:] = torch.randint(0, 97, (2, 5))
    with torch.no_grad():
        a, b = m(idx), m(idx2)
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=1e-10, atol=1e-10)
    assert (a[:, 7:] - b[:, 7:]).abs().max() > 1e-6


@pytest.mark.parametrize("attn,mlp,pkg", COMBOS)
def test_document_isolation_and_position_reset(attn, mlp, pkg):
    m = tiny(attn, mlp, pkg)
    d0, d1 = torch.randint(0, 97, (1, 5)), torch.randint(0, 97, (1, 6))
    packed = torch.cat([d0, d1], 1)
    doc = torch.tensor([[0] * 5 + [1] * 6])
    with torch.no_grad():
        full = m(packed, doc)
        alone = m(d1)
    torch.testing.assert_close(full[:, 5:], alone, rtol=1e-10, atol=1e-10)


def test_param_counts_match_brief():
    def count(attn, mlp, d_mlp, layers=12, d=512, heads=8):
        cfg = ModelConfig(layers=layers, d_model=d, heads=heads, d_mlp=d_mlp, attention=attn, mlp=mlp)
        with torch.device("meta"):
            return TensorLM.__new__(TensorLM), cfg
    # meta-device construction skips init cost; compute from shapes directly
    from tnlm.model import Block
    def matrices(attn, mlp, d_mlp, layers, d, heads):
        cfg = ModelConfig(layers=layers, d_model=d, heads=heads, d_mlp=d_mlp, attention=attn, mlp=mlp)
        with torch.device("meta"):
            blk = Block(cfg)
        per = sum(p.numel() for p in blk.parameters() if p.dim() >= 2)
        return layers * per + 2 * cfg.padded_vocab * d
    assert matrices("softmax", "swiglu", 2048, 12, 512, 8) == 101_842_944  # 101.84M
    assert matrices("signed_product", "bilinear", 1664, 12, 512, 8) == 101_056_512  # 101.06M
    assert round(matrices("softmax", "swiglu", 4096, 24, 1024, 16) / 1e6, 2) == 505.68
    assert round(matrices("signed_product", "bilinear", 3392, 24, 1024, 16) / 1e6, 2) == 504.10


def test_operator_audit():
    assert tiny("softmax", "swiglu", "pre_rms").operator_audit()["class"] == "G"
    assert tiny("signed_product", "bilinear", "pre_rms").operator_audit()["class"] == "N"
    assert tiny("signed_product", "bilinear", "lagged").operator_audit()["class"] == "P"
    with pytest.raises(ValueError):
        ModelConfig(normalization="lagged", final_normalization="rms")


# ---------------------------------------------------------------- lagged / spectral state
def test_lagged_uses_previous_statistic_and_commits_once():
    n = LaggedRMS(4, eps=0.0, half_life_tokens=10.0).double()
    x = torch.randn(3, 4, dtype=D64) * 7
    n.collecting = True
    y = n(x)
    torch.testing.assert_close(y, x)  # mean_sq starts at 1 -> current batch did not affect its output
    n(x)  # second microbatch accumulates, does not update
    assert float(n.mean_sq) == 1.0
    n.commit()  # first commit calibrates
    torch.testing.assert_close(n.mean_sq, x.pow(2).mean())
    ms0 = float(n.mean_sq)
    n(x * 0)
    n.commit()  # 3 tokens, half-life 10 tokens
    keep = 0.5 ** (3 / 10)
    assert abs(float(n.mean_sq) - keep * ms0) < 1e-12


def test_lagged_model_classification_and_buffers():
    m = tiny("signed_product", "bilinear", "lagged")
    assert len(lagged_modules(m)) == 2 * 2 + 1
    assert all("mean_sq" in k or "calibrated" in k or not k.endswith(("acc_sum", "acc_count"))
               for k in m.state_dict())


def test_spectral_effective_weight_reproduces_logits():
    m = tiny("signed_product", "bilinear", "spectral_rms")
    lin = m.blocks[0].attn.q1
    assert isinstance(lin, SpectralLinear)
    idx = torch.randint(0, 97, (1, 8))
    with torch.no_grad():
        ref = m(idx)
        assert freeze_spectral(m) == 2 * 9  # q1 k1 q2 k2 v o l r d per layer
        torch.testing.assert_close(m(idx), ref)


# ---------------------------------------------------------------- padding / empty rows
def test_padding_is_not_a_shared_document():
    doc = torch.tensor([[0, 0, -1, -1]])
    vis = visible_mask(doc)[0, 0]
    assert not vis[2:].any() and not vis[:, 2:].any()
    q1, k1, q2, k2, v = [t.detach() for t in _qkv(B=1, T=4)]
    for kind, p in (("signed_product", 0.5), ("offset_square", 1.0), ("lower_degree_mix", 0.5)):
        C = row_scale_mask(visible_mask(doc), p, D64)
        y = polynomial_attention(kind, q1, k1, q2, k2, v, C, coef=torch.ones(3, 4, dtype=D64),
                                 b=torch.ones(3, dtype=D64))
        assert y[:, :, 2:].abs().max() == 0
    from tnlm.attention import softmax_attention
    y = softmax_attention(q1, k1, v, visible_mask(doc))
    assert torch.isfinite(y).all() and y[:, :, 2:].abs().max() == 0


@pytest.mark.parametrize("attn,mlp,pkg", COMBOS)
def test_padding_values_do_not_affect_real_tokens(attn, mlp, pkg):
    m = tiny(attn, mlp, pkg)
    idx = torch.randint(0, 97, (1, 10))
    doc = torch.tensor([[0] * 7 + [-1] * 3])
    idx2 = idx.clone(); idx2[:, 7:] = torch.randint(0, 97, (1, 3))
    with torch.no_grad():
        a, b = m(idx, doc), m(idx2, doc)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------- spectral edge cases
def test_spectral_zero_weight_is_finite_and_zero():
    lin = SpectralLinear(5, 4).double()
    lin.weight.data.zero_()
    lin.reset_gamma()
    for _ in range(3):
        lin._power_step()
    W = lin.effective_weight()
    assert torch.isfinite(W).all() and W.abs().max() == 0
    assert torch.isfinite(lin.u).all() and lin.u.norm() > 0.5


def test_spectral_freeze_after_updates_with_affine_biases():
    from tnlm.normalization import spectral_step
    m = tiny("signed_product", "affine_bilinear", "spectral_rms").train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    idx = torch.randint(0, 97, (2, 8))
    for _ in range(3):
        spectral_step(m)
        _, loss = m(idx, targets=idx.roll(-1, 1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        ref = m(idx)
        assert freeze_spectral(m) == 2 * 9  # q1 k1 q2 k2 v o l r d per layer
        torch.testing.assert_close(m(idx), ref)


# ---------------------------------------------------------------- extraction from actual modules
@pytest.mark.parametrize("mlp", ["bilinear", "affine_bilinear"])
def test_actual_module_tensor_extraction_full_basis(mlp):
    from tnlm.analysis import mlp_local_tensor
    m = tiny("signed_product", mlp, "pre_rms")
    for p in m.parameters():  # non-trivial gains and biases
        if p.dim() == 1:
            p.data.normal_()
    blk = m.blocks[1]
    d = m.cfg.d_model
    x = torch.randn(5, d, dtype=D64)
    T = mlp_local_tensor(blk, torch.eye(d, dtype=D64), fold_gain=True).double()
    xa = torch.cat([x, torch.ones(5, 1, dtype=D64)], 1) if T.shape[1] > d else x
    # folded tensor applied to x/r reproduces MLP(norm(x))
    r = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + blk.mlp_in.eps)
    xs = torch.cat([x / r, torch.ones(5, 1, dtype=D64)], 1) if T.shape[1] > d else x / r
    with torch.no_grad():
        ref = blk.mlp(blk.mlp_in(x))
    torch.testing.assert_close(torch.einsum("oab,na,nb->no", T, xs, xs), ref, rtol=1e-5, atol=1e-6)


def test_gauge_rescale_and_permutation_leave_function_and_forms_invariant():
    from tnlm.analysis import low_rank_structure, mlp_local_tensor
    m = tiny("signed_product", "bilinear", "pre_rms")
    blk = m.blocks[0]
    d = m.cfg.d_model
    x = torch.randn(4, 3, d, dtype=D64)
    with torch.no_grad():
        y0 = blk.mlp(x)
        T0 = mlp_local_tensor(blk, torch.eye(d, dtype=D64))
        c = torch.rand(m.cfg.d_mlp, dtype=D64) * 3 + 0.2
        perm = torch.randperm(m.cfg.d_mlp)
        blk.mlp.l.weight.mul_(c[:, None]); blk.mlp.r.weight.div_(c[:, None])
        blk.mlp.l.weight.copy_(blk.mlp.l.weight[perm]); blk.mlp.r.weight.copy_(blk.mlp.r.weight[perm])
        blk.mlp.d.weight.copy_(blk.mlp.d.weight[:, perm])
        torch.testing.assert_close(blk.mlp(x), y0)
        torch.testing.assert_close(mlp_local_tensor(blk, torch.eye(d, dtype=D64)), T0, rtol=1e-6, atol=1e-6)


def test_homogeneous_cancellation_fails_under_token_mixing():
    """R0(A(R0 X)) != R0(A(X)) for attention: sources are normalized independently (brief §3)."""
    T, d, dk = 5, 8, 4
    W = [torch.randn(dk, d, dtype=D64) for _ in range(4)] + [torch.randn(d, d, dtype=D64)]
    C = row_scale_mask(visible_mask(torch.zeros(1, T, dtype=torch.long)), 0.5, D64)

    def A(X):
        q1, k1, q2, k2 = (X @ w.T for w in W[:4])
        return polynomial_attention("signed_product", q1[None, None], k1[None, None], q2[None, None],
                                    k2[None, None], (X @ W[4].T)[None, None], C)[0, 0]
    X = torch.randn(T, d, dtype=D64) * torch.tensor([0.3, 1, 4, 0.5, 2], dtype=D64)[:, None]
    assert (r0(A(r0(X))) - r0(A(X))).abs().max() > 1e-3
    # equal per-token norms remove the source reweighting: then the identity holds
    Xe = r0(X)
    torch.testing.assert_close(r0(A(r0(Xe * 3))), r0(A(Xe * 3)))


# ---------------------------------------------------------------- lagged statistics
def test_lagged_excludes_padding_and_splits_microbatches_exactly():
    x = torch.randn(4, 6, 8, dtype=D64)
    valid = torch.ones(4, 6, dtype=D64); valid[:, 4:] = 0
    x_pad = x.clone(); x_pad[:, 4:] = 1e6

    def run(chunks):
        n = LaggedRMS(8).double()
        n.collecting = True
        for xs, vs in chunks:
            n.set_valid_mask(vs); n(xs)
        n.commit()
        return float(n.mean_sq)
    whole = run([(x_pad, valid)])
    split = run([(x_pad[:1], valid[:1]), (x_pad[1:], valid[1:])])
    assert abs(whole - split) < 1e-12
    assert abs(whole - float(x[:, :4].pow(2).mean())) < 1e-12


def test_lagged_frozen_eval_is_immutable_and_has_range():
    n = LaggedRMS(4).double()
    n.mean_sq.fill_(2.0)
    x = torch.randn(3, 4, dtype=D64)
    for _ in range(3):
        n(x)  # collecting is False
    assert float(n.acc_count) == 0
    n.commit()
    assert float(n.mean_sq) == 2.0
    n.mean_sq.fill_(1e60)
    y = n(torch.ones(2, 4, dtype=torch.float32))
    assert torch.isfinite(y).all() and abs(float(y[0, 0]) - 1e-30) < 1e-36


# ---------------------------------------------------------------- memory features: exactness
@pytest.mark.parametrize("pkg", ["pre_rms", "spectral_rms"])
def test_fused_ce_and_checkpointing_match_reference_grads(pkg):
    torch.manual_seed(0)
    base = tiny("signed_product", "affine_bilinear", pkg).train()
    other = tiny("signed_product", "affine_bilinear", pkg, loss_chunk_tokens=7,
                 activation_checkpointing=True).train()
    other.load_state_dict(base.state_dict())
    idx = torch.randint(0, 97, (2, 12))
    doc = torch.tensor([[0] * 5 + [1] * 7, [0] * 9 + [-1] * 3])
    tgt = idx.roll(-1, 1).masked_fill(doc < 0, -100)
    _, la, na = base(idx, doc, tgt, reduction="sum"); la.backward()
    lo, lb, nb = other(idx, doc, tgt, reduction="sum"); lb.backward()
    assert lo is None and int(na) == int(nb)
    torch.testing.assert_close(lb, la, rtol=1e-10, atol=1e-10)
    for (n, a), (_, b) in zip(base.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-8, atol=1e-10, msg=n)


def test_analysis_modes_cannot_leak_into_training():
    m = tiny("signed_product", "bilinear", "pre_rms").train()
    idx = torch.randint(0, 97, (1, 6))
    m.blocks[0].ablate = frozenset({"mlp"})
    with pytest.raises(RuntimeError):
        m(idx)
    m.blocks[0].ablate = frozenset()
    m.blocks[0].attn_in.denom_mode = "record"
    with pytest.raises(RuntimeError):
        m(idx)


def test_depth_residual_scaling_identity_at_base_and_single_convention():
    a = tiny("signed_product", "bilinear", "pre_rms")
    b = tiny("signed_product", "bilinear", "pre_rms", depth_residual_base=2)  # layers == base -> identical
    torch.testing.assert_close(a.blocks[0].mlp.d.weight, b.blocks[0].mlp.d.weight)
    assert a.blocks[0].alpha == b.blocks[0].alpha == 1.0
    cfg = ModelConfig(**{**TINY, "layers": 4}, depth_residual_base=2)
    torch.manual_seed(1)
    m = TensorLM(cfg).double()
    assert m.blocks[0].alpha == 0.5
    # D/O init uses the base depth (not layers): std ~ (2 * 2 * fan_in)^-1/2
    w = m.blocks[0].mlp.d.weight
    assert abs(float(w.std()) - (2 * 2 * w.shape[1]) ** -0.5) < 0.2 * (2 * 2 * w.shape[1]) ** -0.5


# ---------------------------------------------------------------- analysis audit fixes (Codex review 2026-10-16)
def _probe(idx, doc, tgt):
    return [(idx, doc, tgt)]


def test_fixed_denominator_reports_counts_and_skips_empty_targets():
    from tnlm.analysis import fixed_denominator
    m = tiny("signed_product", "bilinear", "pre_rms")
    idx = torch.randint(0, 97, (2, 8)); doc = torch.zeros(2, 8, dtype=torch.long)
    good = (idx, doc, idx.roll(-1, 1))
    empty = (idx, doc, torch.full_like(idx, -100))
    r = fixed_denominator(m, [good, empty])
    assert r["skipped"] == "no scored evaluation targets" and r["evaluation_targets"] == 0
    r = fixed_denominator(m, [good, good])
    assert r["fit_eval_provenance"].startswith("overlapping") and r["evaluation_targets"] == 16
    idx2 = torch.randint(0, 97, (2, 8))
    r = fixed_denominator(m, [good, (idx2, doc, idx2.roll(-1, 1))])
    assert r["fit_eval_provenance"].startswith("row-disjoint") and math.isfinite(r["kl"])


def test_fixed_denominator_skip_does_not_claim_polynomial_class():
    from tnlm.analysis import fixed_denominator
    m = tiny("softmax", "swiglu", "lagged")
    idx = torch.randint(0, 97, (1, 6))
    r = fixed_denominator(m, [(idx, torch.zeros_like(idx), idx)])
    assert r["operator_class"] == "G" and "class P" not in r["skipped"]


def test_lagged_divisor_is_folded_and_gauge_invariant():
    from tnlm.analysis import gain, mlp_local_tensor
    m = tiny("signed_product", "bilinear", "lagged")
    norm = m.blocks[0].mlp_in
    norm.mean_sq.fill_(2.0)
    d = m.cfg.d_model
    T0 = mlp_local_tensor(m.blocks[0], torch.eye(d, dtype=D64))
    x = torch.randn(3, 5, d, dtype=D64)
    with torch.no_grad():
        y0 = m.blocks[0].mlp(norm(x))
        c = 3.0  # exact gauge: g -> c g, mean_sq -> c^2 (mean_sq + eps) - eps
        norm.weight.mul_(c); norm.mean_sq.copy_(c ** 2 * (norm.mean_sq + norm.eps) - norm.eps)
        torch.testing.assert_close(m.blocks[0].mlp(norm(x)), y0)
    torch.testing.assert_close(mlp_local_tensor(m.blocks[0], torch.eye(d, dtype=D64)), T0)


def test_read_write_layer_window_bounds_pairs():
    from tnlm.analysis import read_write_graph
    m = TensorLM(ModelConfig(**{**TINY, "layers": 4})).double()
    full, _ = read_write_graph(m)
    win, _ = read_write_graph(m, layer_window=1)
    assert win["pairs"] < full["pairs"] and win["layer_window"] == 1
    assert full["max_abs_cos"] >= win["max_abs_cos"]


def test_stability_stats_shapes_and_values():
    from tnlm.analysis import stability_stats
    m = tiny("signed_product", "bilinear", "pre_rms")
    idx = torch.randint(0, 97, (2, 12)); doc = torch.tensor([[0] * 5 + [1] * 7, [0] * 12])
    r = stability_stats(m, idx, doc)
    assert len(r["per_layer"]) == 2 and r["tokens"] == 12
    L0 = r["per_layer"][0]
    assert L0["S1S2_max_abs"] <= L0["S1_max_abs"] * L0["S2_max_abs"] + 1e-9
    assert all(math.isfinite(v) for v in r["worst"].values())
    r_soft = stability_stats(tiny("softmax", "swiglu", "pre_rms"), idx, doc)
    assert "S1_max_abs" not in r_soft["per_layer"][0] and "mlp_prod_rms" not in r_soft["per_layer"][0]


@pytest.mark.parametrize("pkg", ["pre_rms", "group_rms", "lagged"])
def test_full_analyze_runs_in_fp64(pkg, tmp_path):
    """Structural guard for the recurring FP64->FP32 downcast bug class: every diagnostic must run on an FP64 model."""
    from tnlm.analysis import analyze
    m = tiny("signed_product", "affine_bilinear" if pkg == "pre_rms" else "bilinear", pkg)
    batches = []
    for _ in range(2):
        idx = torch.randint(0, 97, (2, 16)); doc = torch.tensor([[0] * 6 + [1] * 10, [0] * 12 + [-1] * 4])
        batches.append((idx, doc, idx.roll(-1, 1).masked_fill(doc < 0, -100)))
    r = analyze(m, batches, str(tmp_path))
    assert r["operator_class"] in ("N", "P") and "stability" in r


def test_qk_init_scale_scales_query_key_init_only():
    import torch
    from tnlm.config import ModelConfig
    from tnlm.model import TensorLM
    kw = dict(layers=2, d_model=32, heads=2, d_key=16, d_value=16, d_mlp=48, vocab_size=97, padded_vocab=128)
    torch.manual_seed(0); a = TensorLM(ModelConfig(**kw))
    torch.manual_seed(0); b = TensorLM(ModelConfig(**kw, qk_init_scale=0.1))
    for n, p in a.named_parameters():
        q = dict(b.named_parameters())[n]
        if any(t in n for t in (".q1.", ".k1.", ".q2.", ".k2.")):
            torch.testing.assert_close(q, 0.1 * p)
        else:
            torch.testing.assert_close(q, p)
