"""Acceptance tests for the Triton polynomial attention against the dense reference (brief §8).

Tolerances are fixed here, before any optimization: FP32 (IEEE dots) rel 2e-4 on outputs and all
gradients; BF16 checked relative to the reference magnitude distribution.
"""
import pytest
import torch

from tnlm.attention import doc_positions, polynomial_attention, row_scale_mask, visible_mask

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def _inputs(B, H, T, dk, dv, dtype, docs=None, seed=0, layout="bhtd"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "bthd":  # non-contiguous (B,H,T,d) views of (B,T,H,d) storage, as produced by the model
        mk = lambda d: torch.randn(B, T, H, d, device="cuda", dtype=dtype, generator=g).transpose(1, 2).requires_grad_()
    else:
        mk = lambda d: torch.randn(B, H, T, d, device="cuda", dtype=dtype, generator=g).requires_grad_()
    q1, k1, q2, k2, v = mk(dk), mk(dk), mk(dk), mk(dk), mk(dv)
    if docs is None:
        docs = torch.zeros(B, T, dtype=torch.long, device="cuda")
    return q1, k1, q2, k2, v, docs


def _scale(docs, p):
    pos = doc_positions(docs)
    return torch.where(docs >= 0, (pos + 1).float().pow(-p), torch.zeros((), device=docs.device))


def _params(kind, H):
    coef = torch.tensor([[0.3, -0.5, 0.7, 1.1]], device="cuda").repeat(H, 1).requires_grad_()
    b = torch.linspace(-1, 1.5, H, device="cuda").requires_grad_()
    return coef, b


def _run_both(kind, q1, k1, q2, k2, v, docs, coef, b):
    from tnlm.kernels.poly_attn import poly_attention_triton
    p = 1.0 if kind == "offset_square" else 0.5
    C = row_scale_mask(visible_mask(docs), p)
    ref_in = [t.detach().double().requires_grad_() for t in (q1, k1, q2, k2, v, coef, b)]
    ref = polynomial_attention(kind, *ref_in[:5], C.double(), coef=ref_in[5], b=ref_in[6])
    out = poly_attention_triton(kind, q1, k1, q2, k2, v, docs, _scale(docs, p), coef=coef, b=b)
    go = torch.randn_like(ref)
    ref.backward(go)
    out.backward(go.to(out.dtype))
    names = ["q1", "k1", "q2", "k2", "v", "coef", "b"]
    used = {"signed_product": [0, 1, 2, 3, 4], "offset_square": [0, 1, 4, 6], "lower_degree_mix": [0, 1, 2, 3, 4, 5]}[kind]
    tri = [q1, k1, q2, k2, v, coef, b]
    errs = {"out": _rel(out, ref)}
    for i in used:
        errs[names[i]] = _rel(tri[i].grad, ref_in[i].grad)
    return errs


@cuda
@pytest.mark.parametrize("kind", ["signed_product", "offset_square", "lower_degree_mix"])
@pytest.mark.parametrize("B,H,T,dk,dv", [(2, 3, 128, 64, 64), (1, 2, 77, 32, 64), (2, 1, 200, 64, 32), (1, 1, 5, 16, 16),
                                         (1, 2, 96, 128, 128)])
def test_fp32_matches_reference(kind, B, H, T, dk, dv):
    q1, k1, q2, k2, v, docs = _inputs(B, H, T, dk, dv, torch.float32)
    coef, b = _params(kind, H)
    errs = _run_both(kind, q1, k1, q2, k2, v, docs, coef, b)
    assert max(errs.values()) < 2e-4, errs


@cuda
@pytest.mark.parametrize("kind", ["signed_product", "offset_square", "lower_degree_mix"])
def test_documents_padding_and_layout(kind):
    B, H, T = 2, 2, 150
    docs = torch.zeros(B, T, dtype=torch.long, device="cuda")
    docs[0, 40:] = 1; docs[0, 97:] = 2; docs[0, 140:] = -1  # three docs + padding
    docs[1, 1:] = 1; docs[1, 2:] = 2
    q1, k1, q2, k2, v, _ = _inputs(B, H, T, 64, 64, torch.float32, layout="bthd")
    coef, b = _params(kind, H)
    errs = _run_both(kind, q1, k1, q2, k2, v, docs, coef, b)
    assert max(errs.values()) < 2e-4, errs


@cuda
def test_padding_rows_are_zero_and_large_inputs_finite():
    from tnlm.kernels.poly_attn import poly_attention_triton
    B, H, T = 1, 1, 70
    docs = torch.zeros(B, T, dtype=torch.long, device="cuda"); docs[:, 60:] = -1
    q1, k1, q2, k2, v, _ = _inputs(B, H, T, 64, 64, torch.float32)
    y = poly_attention_triton("signed_product", q1 * 30, k1 * 30, q2, k2, v, docs, _scale(docs, 0.5))
    assert torch.isfinite(y).all() and y[:, :, 60:].abs().max() == 0
    z = poly_attention_triton("signed_product", q1 * 0, k1, q2, k2, v, docs, _scale(docs, 0.5))
    assert z.abs().max() == 0


@cuda
@pytest.mark.parametrize("kind", ["signed_product", "lower_degree_mix"])
@pytest.mark.parametrize("hd", [64, 128])
def test_bf16_error_relative_to_reference_magnitude(kind, hd):
    q1, k1, q2, k2, v, docs = _inputs(2, 4, 512, hd, hd, torch.bfloat16)
    coef, b = _params(kind, 4)
    errs = _run_both(kind, q1, k1, q2, k2, v, docs, coef, b)
    assert max(errs.values()) < 3e-2, errs


@cuda
@pytest.mark.parametrize("attn,mlp", [("signed_product", "bilinear"), ("offset_square", "bilinear"),
                                      ("lower_degree_mix", "affine_bilinear")])
def test_model_triton_backend_matches_reference(attn, mlp):
    from tnlm.config import ModelConfig
    from tnlm.model import TensorLM
    kw = dict(layers=2, d_model=128, heads=2, d_key=64, d_value=64, d_mlp=96, vocab_size=97, padded_vocab=128,
              attention=attn, mlp=mlp)
    torch.manual_seed(0)
    ref = TensorLM(ModelConfig(**kw, attention_backend="reference")).cuda()
    tri = TensorLM(ModelConfig(**kw, attention_backend="triton")).cuda()
    tri.load_state_dict(ref.state_dict())
    idx = torch.randint(0, 97, (2, 100), device="cuda")
    doc = torch.zeros(2, 100, dtype=torch.long, device="cuda"); doc[0, 30:] = 1; doc[1, 90:] = -1
    tgt = idx.roll(-1, 1).masked_fill(doc < 0, -100)
    _, lr = ref(idx, doc, tgt); lr.backward()
    _, lt = tri(idx, doc, tgt); lt.backward()
    assert abs(lr.item() - lt.item()) < 1e-4 * abs(lr.item())
    for (n, a), (_, b) in zip(ref.named_parameters(), tri.named_parameters()):
        assert _rel(b.grad, a.grad) < 1e-3, n


@cuda
@pytest.mark.parametrize("kind", ["signed_product", "lower_degree_mix"])
def test_many_short_documents_block_skipping(kind):
    """Docs shorter than, equal to and straddling tiles; padding runs; a doc starting exactly on a tile edge."""
    B, H, T = 3, 2, 260
    docs = torch.zeros(B, T, dtype=torch.long, device="cuda")
    bounds = [[0, 5, 37, 64, 65, 128, 200, 250], [0, 32, 96, 160, 224], [0, 1, 2, 3, 130]]
    for b, bs in enumerate(bounds):
        for i, start in enumerate(bs):
            docs[b, start:] = i
    docs[0, 255:] = -1
    docs[2, 140:] = -1
    q1, k1, q2, k2, v, _ = _inputs(B, H, T, 64, 64, torch.float32)
    coef, b = _params(kind, H)
    errs = _run_both(kind, q1, k1, q2, k2, v, docs, coef, b)
    assert max(errs.values()) < 2e-4, errs


@cuda
def test_softmax_control_flex_matches_dense_reference():
    from tnlm.config import ModelConfig
    from tnlm.model import TensorLM
    kw = dict(layers=2, d_model=128, heads=2, d_key=64, d_value=64, d_mlp=256, vocab_size=97, padded_vocab=128,
              attention="softmax", mlp="swiglu")
    torch.manual_seed(0)
    ref = TensorLM(ModelConfig(**kw, attention_backend="reference")).cuda()
    fast = TensorLM(ModelConfig(**kw, attention_backend="triton")).cuda()
    fast.load_state_dict(ref.state_dict())
    idx = torch.randint(0, 97, (2, 256), device="cuda")
    doc = torch.zeros(2, 256, dtype=torch.long, device="cuda"); doc[0, 70:] = 1; doc[0, 200:] = 2; doc[1, 240:] = -1
    tgt = idx.roll(-1, 1).masked_fill(doc < 0, -100)
    _, lr = ref(idx, doc, tgt); lr.backward()
    _, lf = fast(idx, doc, tgt); lf.backward()
    assert abs(lr.item() - lf.item()) < 1e-4 * abs(lr.item())
    for (n, a), (_, b) in zip(ref.named_parameters(), fast.named_parameters()):
        assert _rel(b.grad, a.grad) < 2e-3, n


@cuda
def test_tuning_override_is_used_and_restored():
    from tnlm.kernels import poly_attn as pa
    from tnlm.kernels.tune import sweep
    rows = sweep(head_dim=64, batch=1, heads=2, context=128, doc_len=50, blocks=(32,), stages=(2,), warps=(4,))
    assert len(rows) == 1 and rows[0]["ms"] < float("inf")
    assert pa.TUNING == {}
