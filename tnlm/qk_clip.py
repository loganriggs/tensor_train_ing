"""QK-clip for polynomial attention (scaling_review rec. 7; MuonClip-style weight rescaling).

After an optimizer update, measure each head's max |S| = max |q.k| / sqrt(dk) over a fixed probe (causally visible, same-document
pairs), separately for the S1 (q1,k1) and S2 (q2,k2) factors. If a head's max exceeds `tau`, rescale that head's query and key
projection rows by sqrt(tau / max). That multiplies its scores by tau/max exactly, so the clipped head's max |S| equals tau.

Properties that matter for this project:
- It only rescales weights, so the model stays exactly polynomial (P/N class unchanged) and every export stays a plain linear map.
- It acts per head and per factor, so S1*S2 is bounded by tau^2 at the probe.
- It's a training-time intervention like weight decay. It must be applied at a fixed cadence and logged (returned per-head factors).
- Probe-based: max |S| on unseen data can exceed tau. The bound holds on the probe, not globally.
- Sequential: layers are clipped in order with re-measurement, because clipping layer i changes layer i+1's inputs.
"""
from __future__ import annotations

import torch

from .normalization import SpectralLinear


@torch.no_grad()
def head_score_max(model, idx, doc) -> list[dict]:
    """Per layer: {'S1': (H,), 'S2': (H,)} max |S| over visible pairs of the probe (first row); {} for softmax layers."""
    from .analysis import probe_attention_scores
    scores, _, vis = probe_attention_scores(model, idx, doc, max_tokens=idx.shape[1])
    return [{k: S.masked_fill(~vis, 0).abs().amax(dim=(-1, -2)) for k, S in row.items()} for row in scores]


@torch.no_grad()
def _rescale_heads_(blk, name: str, gamma: torch.Tensor, d_key: int):
    """Multiply factor `name` ('S1'/'S2') of each head's scores by gamma (H,): scale its Q and K rows by sqrt(gamma)."""
    q, k = (blk.attn.q1, blk.attn.k1) if name == "S1" else (blk.attn.q2, blk.attn.k2)
    rows = gamma.sqrt().repeat_interleave(d_key).to(q.weight.dtype)  # (H*dk,)
    for lin in (q, k):
        lin.weight.mul_(rows[:, None])
        if lin.bias is not None:
            lin.bias.mul_(rows)


@torch.no_grad()
def qk_clip_(model, idx, doc, tau: float) -> dict:
    """Clip in place. Returns {'clipped_heads': n, 'max_before': float, 'factors': {layer: {S1: [...], S2: [...]}}}."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if any(isinstance(m, SpectralLinear) for m in model.modules()):
        # W_eff = gamma * W / sigma(W): row rescaling of W changes sigma, so the clip would not be exact
        raise NotImplementedError("qk_clip_ is exact only for plain linear Q/K (not spectral_rms)")
    cfg = model.cfg
    n, worst, factors = 0, 0.0, {}
    # Layer by layer: clipping layer i changes the residual stream read by later layers, so their scores are
    # re-measured after each clip (one probe forward per layer; called at a sparse cadence).
    for i, blk in enumerate(model.blocks):
        st = head_score_max(model, idx, doc)[i]
        for name, mx in st.items():
            worst = max(worst, float(mx.max()))
            gamma = torch.clamp(tau / mx.clamp_min(1e-30), max=1.0)  # (H,)
            if bool((gamma < 1).any()):
                _rescale_heads_(blk, name, gamma, cfg.d_key)
                n += int((gamma < 1).sum())
                factors.setdefault(i, {})[name] = [float(g) for g in gamma]
    return {"clipped_heads": n, "max_before": worst, "factors": factors, "tau": tau}


def set_score_bound_recording(model, on: bool = True):
    """Toggle per-batch score-bound recording on every polynomial attention block (see tnlm.model.score_bound)."""
    for blk in model.blocks:
        blk.attn.record_score_bound = on
        if not on:
            blk.attn.last_score_bound = None


def score_bounds(model) -> list[dict]:
    """Per layer {'S1': (H,), 'S2': (H,)} upper bounds from the most recent forward with recording on ({} if none)."""
    return [dict(blk.attn.last_score_bound or {}) for blk in model.blocks]


@torch.no_grad()
def qk_clip_from_bounds_(model, tau: float) -> dict:
    """Probe-free QK-clip: rescale each head whose recorded per-batch bound exceeds tau by sqrt(tau/bound) on its Q and K rows.
    Since bound >= true max |S| on that batch, every score of that batch is <= tau afterwards (conservative: may over-clip). Uses the
    bounds from the last forward, so call it right after a training forward/step with recording on. Layers are independent here (each
    bound is from the same batch), unlike the probe version, which must re-measure."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if any(isinstance(m, SpectralLinear) for m in model.modules()):
        raise NotImplementedError("qk_clip is exact only for plain linear Q/K (not spectral_rms)")
    cfg, n, worst, factors = model.cfg, 0, 0.0, {}
    for i, (blk, st) in enumerate(zip(model.blocks, score_bounds(model))):
        for name, bound in st.items():
            worst = max(worst, float(bound.max()))
            gamma = torch.clamp(tau / bound.clamp_min(1e-30), max=1.0)
            if bool((gamma < 1).any()):
                _rescale_heads_(blk, name, gamma, cfg.d_key)
                n += int((gamma < 1).sum())
                factors.setdefault(i, {})[name] = [float(g) for g in gamma]
    return {"clipped_heads": n, "max_bound_before": worst, "factors": factors, "tau": tau, "source": "per-batch norm bound"}
