"""Chunked linear + cross-entropy with recomputation (brief §7: bounded logit memory).

Exact (no gradient filtering): forward keeps only per-token log-sum-exp; backward recomputes each
chunk's logits. Peak extra memory is O(chunk * vocab) instead of O(tokens * vocab). Matmuls run in
the autocast dtype when autocast is active (as nn.Linear would), softmax/LSE in FP32, dW in FP32.
"""
from __future__ import annotations

import torch


def _compute_dtype(x):
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, targets, chunk, cdtype):
        N = x.shape[0]
        xc, wc = x.to(cdtype), w.to(cdtype)
        acc = torch.promote_types(cdtype, torch.float32)
        lse = torch.empty(N, device=x.device, dtype=acc)
        total = torch.zeros((), device=x.device, dtype=acc)
        valid = targets != -100
        safe_t = targets.clamp_min(0)
        for s in range(0, N, chunk):
            lg = (xc[s:s + chunk] @ wc.T).to(acc)
            l = torch.logsumexp(lg, -1)
            lse[s:s + chunk] = l
            tl_ = lg.gather(1, safe_t[s:s + chunk, None])[:, 0]
            total += ((l - tl_) * valid[s:s + chunk]).sum()
        ctx.save_for_backward(x, w, targets, lse)
        ctx.chunk, ctx.cdtype = chunk, cdtype
        return total

    @staticmethod
    def backward(ctx, g):
        x, w, targets, lse = ctx.saved_tensors
        chunk, cdtype = ctx.chunk, ctx.cdtype
        xc, wc = x.to(cdtype), w.to(cdtype)
        dx = torch.empty_like(x)
        acc = lse.dtype
        dw = torch.zeros(w.shape, device=w.device, dtype=acc)
        valid = targets != -100
        safe_t = targets.clamp_min(0)
        for s in range(0, x.shape[0], chunk):
            sl = slice(s, s + chunk)
            p = torch.exp((xc[sl] @ wc.T).to(acc) - lse[sl, None])
            p.scatter_add_(1, safe_t[sl, None], -torch.ones_like(p[:, :1]))
            p = (p * (valid[sl, None] * g)).to(cdtype)
            dx[sl] = (p @ wc).to(x.dtype)
            dw += (p.T @ xc[sl]).to(acc)
        return dx, dw.to(w.dtype), None, None, None


def linear_cross_entropy_sum(x, w, targets, chunk=4096):
    """sum over valid targets of CE(x @ w.T, target); x: (N, d), w: (V, d), targets: (N,) with -100 ignored."""
    return LinearCrossEntropy.apply(x, w, targets, chunk, _compute_dtype(x))
