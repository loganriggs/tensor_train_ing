"""Tiled causal polynomial attention in Triton: forward and backward (brief §8).

Computes, without materializing any T x T matrix,
    y_i = scale_i * sum_{j visible from i} A(S1_ij, S2_ij) v_j,   S_ij = q_i . k_j / sqrt(dk)
with A one of
    KIND 0 signed_product   : S1 * S2
    KIND 1 offset_square    : (S1 + b_h)^2              (q2/k2 unused)
    KIND 2 lower_degree_mix : c0 + c1 S1 + c2 S2 + c12 S1 S2
"visible" = causal and same document (doc ids >= 0, contiguous per row). The mask is applied
*after* the polynomial (zero, never -inf); there is no row softmax. scale_i = n_visible_i^-p is
precomputed (0 for padding queries), so the mask/scale enters each derivative exactly once.

Precision policy: scores and their products are formed in FP32 from the input dtype; the
masked product P is cast to v's dtype for the P @ V tensor-core matmul; all accumulators are FP32.
Learned head scalars (b, coef) get gradients via deterministic per-block partial sums.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

KINDS = {"signed_product": 0, "offset_square": 1, "lower_degree_mix": 2}


@triton.jit
def _poly(s1, s2, c0, c1, c2, c12, KIND: tl.constexpr):
    if KIND == 0:
        return s1 * s2
    elif KIND == 1:
        t = s1 + c0
        return t * t
    else:
        return c0 + c1 * s1 + c2 * s2 + c12 * s1 * s2


@triton.jit
def _fwd(Q1, K1, Q2, K2, V, DOC, SCALE, PAR, O, LO,
         sq_b, sq_h, sq_t, sv_b, sv_h, sv_t, H, T, NMB, sm,
         KIND: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    mb, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_m = mb * BM + tl.arange(0, BM)
    offs_k, offs_v = tl.arange(0, DK), tl.arange(0, DV)
    mm = offs_m < T
    qbase = b * sq_b + h * sq_h
    vbase = b * sv_b + h * sv_h
    q1 = tl.load(Q1 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], mask=mm[:, None], other=0.)
    if KIND != 1:
        q2 = tl.load(Q2 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], mask=mm[:, None], other=0.)
    dq = tl.load(DOC + b * T + offs_m, mask=mm, other=-2)
    sc = tl.load(SCALE + b * T + offs_m, mask=mm, other=0.)
    c0 = tl.load(PAR + h * 4 + 0); c1 = tl.load(PAR + h * 4 + 1)
    c2 = tl.load(PAR + h * 4 + 2); c12 = tl.load(PAR + h * 4 + 3)
    acc = tl.zeros((BM, DV), dtype=tl.float32)
    hi = tl.minimum((mb + 1) * BM, T)
    lo = (tl.load(LO + b * NMB + mb) // BN) * BN  # earliest document start in this query block
    for start in range(lo, hi, BN):
        offs_n = start + tl.arange(0, BN)
        nm = offs_n < T
        k1 = tl.load(K1 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], mask=nm[:, None], other=0.)
        s1 = tl.dot(q1, tl.trans(k1), input_precision=PREC) * sm
        if KIND != 1:
            k2 = tl.load(K2 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], mask=nm[:, None], other=0.)
            s2 = tl.dot(q2, tl.trans(k2), input_precision=PREC) * sm
        else:
            s2 = s1
        dk = tl.load(DOC + b * T + offs_n, mask=nm, other=-3)
        vis = (offs_m[:, None] >= offs_n[None, :]) & (dq[:, None] == dk[None, :]) & nm[None, :]
        p = tl.where(vis, _poly(s1, s2, c0, c1, c2, c12, KIND), 0.) * sc[:, None]
        v = tl.load(V + vbase + offs_n[:, None] * sv_t + offs_v[None, :], mask=nm[:, None], other=0.)
        acc += tl.dot(p.to(v.dtype), v, input_precision=PREC)
    tl.store(O + vbase + offs_m[:, None] * sv_t + offs_v[None, :], acc.to(O.dtype.element_ty), mask=mm[:, None])


@triton.jit
def _bwd_kv(Q1, K1, Q2, K2, V, DO, DOC, SCALE, PAR, DK1, DK2, DVv, DPAR, HI,
            sq_b, sq_h, sq_t, sv_b, sv_h, sv_t, H, T, NB, sm,
            KIND: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    nb, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_n = nb * BN + tl.arange(0, BN)
    offs_k, offs_v = tl.arange(0, DK), tl.arange(0, DV)
    nm = offs_n < T
    qbase = b * sq_b + h * sq_h
    vbase = b * sv_b + h * sv_h
    k1 = tl.load(K1 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], mask=nm[:, None], other=0.)
    if KIND != 1:
        k2 = tl.load(K2 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], mask=nm[:, None], other=0.)
    v = tl.load(V + vbase + offs_n[:, None] * sv_t + offs_v[None, :], mask=nm[:, None], other=0.)
    dk = tl.load(DOC + b * T + offs_n, mask=nm, other=-3)
    c0 = tl.load(PAR + h * 4 + 0); c1 = tl.load(PAR + h * 4 + 1)
    c2 = tl.load(PAR + h * 4 + 2); c12 = tl.load(PAR + h * 4 + 3)
    dk1 = tl.zeros((BN, DK), dtype=tl.float32)
    dk2 = tl.zeros((BN, DK), dtype=tl.float32)
    dvv = tl.zeros((BN, DV), dtype=tl.float32)
    g0 = tl.zeros((BN,), dtype=tl.float32)
    g1 = tl.zeros((BN,), dtype=tl.float32)
    g2 = tl.zeros((BN,), dtype=tl.float32)
    g12 = tl.zeros((BN,), dtype=tl.float32)
    lo = (nb * BN // BM) * BM
    hi = tl.load(HI + b * NB + nb)  # exclusive end of the last document touching this key block
    for start in range(lo, hi, BM):
        offs_m = start + tl.arange(0, BM)
        mm = offs_m < T
        q1 = tl.load(Q1 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], mask=mm[:, None], other=0.)
        s1 = tl.dot(k1, tl.trans(q1), input_precision=PREC) * sm  # (BN, BM) = S1^T
        if KIND != 1:
            q2 = tl.load(Q2 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], mask=mm[:, None], other=0.)
            s2 = tl.dot(k2, tl.trans(q2), input_precision=PREC) * sm
        else:
            s2 = s1
        do = tl.load(DO + vbase + offs_m[:, None] * sv_t + offs_v[None, :], mask=mm[:, None], other=0.)
        dq = tl.load(DOC + b * T + offs_m, mask=mm, other=-2)
        sc = tl.load(SCALE + b * T + offs_m, mask=mm, other=0.)
        vis = (offs_m[None, :] >= offs_n[:, None]) & (dq[None, :] == dk[:, None]) & mm[None, :]
        cm = tl.where(vis, 1., 0.) * sc[None, :]  # C^T
        pT = cm * _poly(s1, s2, c0, c1, c2, c12, KIND)
        dvv += tl.dot(pT.to(do.dtype), do, input_precision=PREC)
        hT = cm * tl.dot(v, tl.trans(do), input_precision=PREC)  # (C * dO V^T)^T
        if KIND == 0:
            ds1 = hT * s2
            ds2 = hT * s1
        elif KIND == 1:
            ds1 = 2. * hT * (s1 + c0)
            g0 += tl.sum(ds1, 1)
        else:
            ds1 = hT * (c1 + c12 * s2)
            ds2 = hT * (c2 + c12 * s1)
            g0 += tl.sum(hT, 1)
            g1 += tl.sum(hT * s1, 1)
            g2 += tl.sum(hT * s2, 1)
            g12 += tl.sum(hT * s1 * s2, 1)
        dk1 += tl.dot(ds1.to(q1.dtype), q1, input_precision=PREC)
        if KIND != 1:
            dk2 += tl.dot(ds2.to(q2.dtype), q2, input_precision=PREC)
    tl.store(DK1 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], (dk1 * sm).to(DK1.dtype.element_ty), mask=nm[:, None])
    if KIND != 1:
        tl.store(DK2 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], (dk2 * sm).to(DK2.dtype.element_ty), mask=nm[:, None])
    tl.store(DVv + vbase + offs_n[:, None] * sv_t + offs_v[None, :], dvv.to(DVv.dtype.element_ty), mask=nm[:, None])
    if KIND != 0:
        pbase = DPAR + (bh * NB + nb) * 4
        tl.store(pbase + 0, tl.sum(g0, 0))
        tl.store(pbase + 1, tl.sum(g1, 0))
        tl.store(pbase + 2, tl.sum(g2, 0))
        tl.store(pbase + 3, tl.sum(g12, 0))


@triton.jit
def _bwd_q(Q1, K1, Q2, K2, V, DO, DOC, SCALE, PAR, DQ1, DQ2, LO,
           sq_b, sq_h, sq_t, sv_b, sv_h, sv_t, H, T, NMB, sm,
           KIND: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    mb, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    offs_m = mb * BM + tl.arange(0, BM)
    offs_k, offs_v = tl.arange(0, DK), tl.arange(0, DV)
    mm = offs_m < T
    qbase = b * sq_b + h * sq_h
    vbase = b * sv_b + h * sv_h
    q1 = tl.load(Q1 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], mask=mm[:, None], other=0.)
    if KIND != 1:
        q2 = tl.load(Q2 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], mask=mm[:, None], other=0.)
    do = tl.load(DO + vbase + offs_m[:, None] * sv_t + offs_v[None, :], mask=mm[:, None], other=0.)
    dq = tl.load(DOC + b * T + offs_m, mask=mm, other=-2)
    sc = tl.load(SCALE + b * T + offs_m, mask=mm, other=0.)
    c0 = tl.load(PAR + h * 4 + 0); c1 = tl.load(PAR + h * 4 + 1)
    c2 = tl.load(PAR + h * 4 + 2); c12 = tl.load(PAR + h * 4 + 3)
    dq1 = tl.zeros((BM, DK), dtype=tl.float32)
    dq2 = tl.zeros((BM, DK), dtype=tl.float32)
    hi = tl.minimum((mb + 1) * BM, T)
    lo = (tl.load(LO + b * NMB + mb) // BN) * BN
    for start in range(lo, hi, BN):
        offs_n = start + tl.arange(0, BN)
        nm = offs_n < T
        k1 = tl.load(K1 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], mask=nm[:, None], other=0.)
        s1 = tl.dot(q1, tl.trans(k1), input_precision=PREC) * sm
        if KIND != 1:
            k2 = tl.load(K2 + qbase + offs_n[:, None] * sq_t + offs_k[None, :], mask=nm[:, None], other=0.)
            s2 = tl.dot(q2, tl.trans(k2), input_precision=PREC) * sm
        else:
            s2 = s1
        v = tl.load(V + vbase + offs_n[:, None] * sv_t + offs_v[None, :], mask=nm[:, None], other=0.)
        dk = tl.load(DOC + b * T + offs_n, mask=nm, other=-3)
        vis = (offs_m[:, None] >= offs_n[None, :]) & (dq[:, None] == dk[None, :]) & nm[None, :]
        hm = tl.where(vis, 1., 0.) * sc[:, None] * tl.dot(do, tl.trans(v), input_precision=PREC)
        if KIND == 0:
            ds1 = hm * s2
            ds2 = hm * s1
        elif KIND == 1:
            ds1 = 2. * hm * (s1 + c0)
        else:
            ds1 = hm * (c1 + c12 * s2)
            ds2 = hm * (c2 + c12 * s1)
        dq1 += tl.dot(ds1.to(k1.dtype), k1, input_precision=PREC)
        if KIND != 1:
            dq2 += tl.dot(ds2.to(k2.dtype), k2, input_precision=PREC)
    tl.store(DQ1 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], (dq1 * sm).to(DQ1.dtype.element_ty), mask=mm[:, None])
    if KIND != 1:
        tl.store(DQ2 + qbase + offs_m[:, None] * sq_t + offs_k[None, :], (dq2 * sm).to(DQ2.dtype.element_ty), mask=mm[:, None])


def _block_bounds(doc, BM, BN):
    """LO[b, mb] = document start of the first query in block mb (contiguous docs => block minimum);
    HI[b, nb] = exclusive end of the document containing the last key of block nb."""
    B, T = doc.shape
    ar = torch.arange(T, device=doc.device).expand(B, T)
    is_start = torch.ones_like(doc, dtype=torch.bool)
    is_start[:, 1:] = doc[:, 1:] != doc[:, :-1]
    start = torch.where(is_start, ar, torch.zeros_like(ar)).cummax(-1).values
    nxt = torch.where(is_start, ar, torch.full_like(ar, T))
    nxt = torch.cat([nxt[:, 1:], torch.full_like(nxt[:, :1], T)], 1).flip(-1).cummin(-1).values.flip(-1)
    lo = start[:, ::BM].to(torch.int32).contiguous()
    last = torch.clamp(torch.arange(BN - 1, T + BN - 1, BN, device=doc.device), max=T - 1)
    hi = nxt[:, last].to(torch.int32).contiguous()
    return lo, hi


def _params(kind, H, coef, b, device):
    par = torch.zeros(H, 4, device=device, dtype=torch.float32)
    if kind == "offset_square":
        par[:, 0] = b.float()
    elif kind == "lower_degree_mix":
        par.copy_(coef.float())
    return par.contiguous()


def _cfg(dtype, T, dmax):
    """(input precision, BM, BN, num_stages, num_warps). Tiles shrink with head dim / dtype to fit ~100KB shared
    memory (Ada/Ampere consumer parts). Re-tune on the target GPU with `python -m tnlm.kernels.tune`; pass the
    winner via TUNING or the STAGES table."""
    prec = "ieee" if dtype == torch.float32 else "tf32"
    key = (dtype != torch.float32, dmax)
    if key in TUNING:
        blk, stages, warps = TUNING[key]
        return prec, min(blk, max(16, triton.next_power_of_2(T))), min(blk, max(16, triton.next_power_of_2(T))), stages, warps
    blk = 64 if (dtype != torch.float32 and dmax <= 64) else 32
    if dtype == torch.float32 and dmax > 64:
        blk = 16
    blk = min(blk, max(16, triton.next_power_of_2(T)))
    return prec, blk, blk, STAGES.get(key, 2), 4


STAGES = {(True, 128): 3}  # (is_low_precision, head_dim) -> num_stages; measured with tnlm.kernels.tune on RTX 5000 Ada
TUNING: dict = {}  # (is_low_precision, head_dim) -> (block, num_stages, num_warps); empty = built-in defaults


def _like(x, ref):
    """x with exactly ref's strides (copy only if needed); kernels index all heads through one stride set."""
    return x if x.stride() == ref.stride() else torch.empty_like(ref).copy_(x)


class PolyAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kind, q1, k1, q2, k2, v, doc_ids, row_scale, coef, b):
        B, H, T, DK = q1.shape
        DV = v.shape[-1]
        K = KINDS[kind]
        assert DK >= 16 and DK & (DK - 1) == 0 and DV >= 16 and DV & (DV - 1) == 0, "head dims: powers of 2 >= 16"
        if q1.stride(-1) != 1:
            q1 = q1.contiguous()
        if v.stride(-1) != 1:
            v = v.contiguous()
        k1 = _like(k1, q1)
        if K != 1:
            q2, k2 = _like(q2, q1), _like(k2, q1)
        else:
            q2, k2 = q1, k1
        doc = doc_ids.to(torch.int32).contiguous()
        sc = row_scale.float().contiguous()
        par = _params(kind, H, coef, b, q1.device)
        o = torch.empty_like(v)
        prec, BM, BN, ST, NW = _cfg(q1.dtype, T, max(DK, DV))
        lo, hi = _block_bounds(doc, BM, BN)
        grid = (triton.cdiv(T, BM), B * H)
        _fwd[grid](q1, k1, q2, k2, v, doc, sc, par, o, lo,
                   q1.stride(0), q1.stride(1), q1.stride(2), v.stride(0), v.stride(1), v.stride(2),
                   H, T, lo.shape[1], DK ** -0.5, KIND=K, DK=DK, DV=DV, BM=BM, BN=BN, PREC=prec, num_warps=NW, num_stages=ST)
        ctx.save_for_backward(q1, k1, q2, k2, v, doc, sc, par, lo, hi)
        ctx.kind, ctx.has_coef, ctx.has_b = kind, coef is not None, b is not None
        return o

    @staticmethod
    def backward(ctx, do):
        q1, k1, q2, k2, v, doc, sc, par, lo, hi = ctx.saved_tensors
        B, H, T, DK = q1.shape
        DV = v.shape[-1]
        K = KINDS[ctx.kind]
        do = _like(do, v)
        prec, BM, BN, ST, NW = _cfg(q1.dtype, T, max(DK, DV))
        NB = triton.cdiv(T, BN)
        dq1, dk1, dv = torch.empty_like(q1), torch.empty_like(k1), torch.empty_like(v)
        dq2, dk2 = (torch.empty_like(q2), torch.empty_like(k2)) if K != 1 else (dq1, dk1)
        dpar = torch.zeros(B * H, NB, 4, device=q1.device, dtype=torch.float32)
        args = (q1.stride(0), q1.stride(1), q1.stride(2), v.stride(0), v.stride(1), v.stride(2), H, T)
        _bwd_kv[(NB, B * H)](q1, k1, q2, k2, v, do, doc, sc, par, dk1, dk2, dv, dpar, hi, *args, NB, DK ** -0.5,
                             KIND=K, DK=DK, DV=DV, BM=BM, BN=BN, PREC=prec, num_warps=NW, num_stages=ST)
        # dQ in its own pass (recomputes S1, S2, dP). A fused single-kernel variant with FP32 atomic dQ was measured
        # 1.3-7.7x SLOWER on RTX 5000 Ada (atomic traffic ~ key-blocks x query-blocks x tile) and was removed; see docs/scaling.md.
        _bwd_q[(triton.cdiv(T, BM), B * H)](q1, k1, q2, k2, v, do, doc, sc, par, dq1, dq2, lo, *args, lo.shape[1], DK ** -0.5,
                                           KIND=K, DK=DK, DV=DV, BM=BM, BN=BN, PREC=prec, num_warps=NW, num_stages=ST)
        g = dpar.view(B, H, NB, 4).sum((0, 2))  # deterministic reduction of per-block partials
        dcoef = g if (K == 2 and ctx.has_coef) else None
        db = g[:, 0] if (K == 1 and ctx.has_b) else None
        if K == 1:
            dq2 = dk2 = None
        return None, dq1, dk1, dq2, dk2, dv, None, None, dcoef, db


def poly_attention_triton(kind, q1, k1, q2, k2, v, doc_ids, row_scale, coef=None, b=None):
    """q*, k*: (B, H, T, dk); v: (B, H, T, dv); doc_ids: (B, T) contiguous ids, <0 = padding;
    row_scale: (B, T) = n_visible^-p for real queries, 0 for padding. Every gradient is reduced in a fixed order."""
    return PolyAttention.apply(kind, q1, k1, q2, k2, v, doc_ids, row_scale, coef, b)
