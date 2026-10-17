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
                q = blk.attn.q1 if name == "S1" else blk.attn.q2
                k = blk.attn.k1 if name == "S1" else blk.attn.k2
                rows = gamma.sqrt().repeat_interleave(cfg.d_key).to(q.weight.dtype)  # (H*dk,)
                for lin in (q, k):
                    lin.weight.mul_(rows[:, None])
                    if lin.bias is not None:
                        lin.bias.mul_(rows)
                n += int((gamma < 1).sum())
                factors.setdefault(i, {})[name] = [float(g) for g in gamma]
    return {"clipped_heads": n, "max_before": worst, "factors": factors, "tau": tau}
