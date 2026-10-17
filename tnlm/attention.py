"""Reference attention (brief §4, §8): one authoritative implementation.

Polynomial variants never use a softmax. Excluded entries are zeroed *after* the
polynomial is formed, and the configured row scale is applied exactly once through C.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def doc_positions(doc_ids: torch.Tensor) -> torch.Tensor:
    """Position within document for contiguous doc_ids of shape (B, T)."""
    B, T = doc_ids.shape
    ar = torch.arange(T, device=doc_ids.device).expand(B, T)
    start = torch.ones_like(doc_ids, dtype=torch.bool)
    start[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    first = torch.where(start, ar, torch.zeros_like(ar)).cummax(-1).values
    return ar - first


def visible_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """(B, 1, T, T) bool: causal, same document, and both query and key are real tokens.

    Negative doc_ids mark padding: padding never attends and is never attended to,
    so its rows are empty (zero output) rather than one shared pseudo-document.
    """
    T = doc_ids.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    same = doc_ids[:, :, None] == doc_ids[:, None, :]
    valid = doc_ids >= 0
    return (causal & same & valid[:, :, None] & valid[:, None, :])[:, None]


def row_scale_mask(visible: torch.Tensor, power: float, dtype=torch.float32) -> torch.Tensor:
    """C = visible * n_visible^-power. Rows with no visible token are all zero."""
    n = visible.sum(-1, keepdim=True).clamp_min(1).to(dtype)
    return visible.to(dtype) * n.pow(-power)


def rope_tables(positions: torch.Tensor, dim: int, base: float):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    ang = positions.float()[..., None] * inv  # (B, T, dim/2)
    cos, sin = ang.cos(), ang.sin()
    return cos[:, None], sin[:, None]  # broadcast over heads: (B, 1, T, dim/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.to(torch.promote_types(x.dtype, torch.float32)).chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


def scores(q, k):
    return q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)


def polynomial_attention(kind: str, q1, k1, q2, k2, v, C, coef=None, b=None):
    """Dense reference. q*, k*: (B, H, T, dk); v: (B, H, T, dv); C: (B, 1|H, T, T).

    signed_product:   A = S1 * S2
    offset_square:    A = (S1 + b)^2          (q2/k2 unused)
    lower_degree_mix: A = c0 + c1 S1 + c2 S2 + c12 S1 S2   (coef: (H, 4))
    """
    S1 = scores(q1, k1)
    if kind == "signed_product":
        A = S1 * scores(q2, k2)
    elif kind == "offset_square":
        A = (S1 + b.view(1, -1, 1, 1)) ** 2
    elif kind == "lower_degree_mix":
        S2 = scores(q2, k2)
        c = coef.view(1, -1, 4, 1, 1).unbind(2)
        A = c[0] + c[1] * S1 + c[2] * S2 + c[3] * S1 * S2
    else:
        raise ValueError(kind)
    P = C * A  # zero masking after the polynomial
    return P.to(v.dtype) @ v


def polynomial_attention_recurrent(q1, k1, q2, k2, v, scale):
    """Exact causal recurrent form of signed_product for a single document (brief §8).

    S_t = S_{t-1} + (k1_t ⊗ k2_t) v_t^T ;  y_t = scale_t * (q1_t ⊗ q2_t)^T S_t
    Shapes (H, T, d). `scale`: (T,) row scale. Includes the 1/dk from two sqrt(dk) score scalings.
    """
    H, T, dk = q1.shape
    dv = v.shape[-1]
    state = q1.new_zeros(H, dk * dk, dv)
    ys = []
    for t in range(T):
        kk = (k1[:, t, :, None] * k2[:, t, None, :]).reshape(H, -1)
        state = state + kk[:, :, None] * v[:, t, None, :]
        qq = (q1[:, t, :, None] * q2[:, t, None, :]).reshape(H, -1)
        ys.append(torch.einsum("hs,hsv->hv", qq, state) * scale[t] / dk)
    return torch.stack(ys, 1)


def softmax_attention(q, k, v, visible):
    """Softmax control. Empty rows (padding queries) output exactly zero instead of NaN."""
    nonempty = visible.any(-1, keepdim=True)
    T = visible.shape[-1]
    eye = torch.eye(T, dtype=torch.bool, device=visible.device)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=visible | (~nonempty & eye))
    return y * nonempty.to(y.dtype)


def softmax_attention_flex(q, k, v, doc_ids):
    """Softmax control with a FlexAttention block mask (causal, same document, no padding).

    Same function as `softmax_attention` on non-empty rows, but O(T) memory, and fully masked
    tiles are skipped, so the control isn't handicapped against the tiled polynomial kernel at long context.
    Padding query rows are zeroed explicitly.
    """
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    B, T = doc_ids.shape
    doc = doc_ids

    def mask_mod(b, h, qi, ki):
        return (qi >= ki) & (doc[b, qi] == doc[b, ki]) & (doc[b, ki] >= 0)

    bm = create_block_mask(mask_mod, B, None, T, T, device=q.device)
    y = _compiled_flex()(q, k, v, block_mask=bm)
    return y * (doc_ids >= 0)[:, None, :, None].to(y.dtype)


_FLEX = None


def _compiled_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX
