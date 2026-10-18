"""Bilinear / polynomial-attention language model and the SwiGLU+softmax control (brief §2-§4)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (apply_rope, doc_positions, polynomial_attention, rope_tables,
                        row_scale_mask, softmax_attention, softmax_attention_flex, visible_mask)
from .config import ModelConfig
from .normalization import GroupRMSNorm, LaggedRMS, RMSNorm, SpectralLinear, make_norm

# package -> (norm_in kind or None, norm_out kind or None, spectral linears)
PACKAGES = {
    "pre_rms": ("rms", None, False),
    "branch_rms": (None, "rms", False),
    "sandwich_rms": ("rms", "rms", False),
    "group_rms": ("group", None, False),
    "spectral_rms": ("rms", None, True),
    "lagged": ("lagged", None, False),
    "taper": ("taper", None, False),
}
CLASS_ORDER = {"P": 0, "N": 1, "G": 2}


def _linear(d_in, d_out, spectral, bias=False):
    return SpectralLinear(d_in, d_out, bias) if spectral else nn.Linear(d_in, d_out, bias=bias)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, spectral: bool):
        super().__init__()
        self.cfg, self.kind = cfg, cfg.attention
        H, dk, dv, d = cfg.heads, cfg.d_key, cfg.d_value, cfg.d_model
        self.q1, self.k1 = _linear(d, H * dk, spectral), _linear(d, H * dk, spectral)
        if self.kind in ("signed_product", "lower_degree_mix"):
            self.q2, self.k2 = _linear(d, H * dk, spectral), _linear(d, H * dk, spectral)
        self.v = _linear(d, H * dv, spectral)
        self.o = _linear(H * dv, d, spectral)
        if self.kind == "offset_square":
            self.b = nn.Parameter(torch.full((H,), cfg.offset_square_b))
        if self.kind == "lower_degree_mix":
            self.coef = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(H, 1))
        self.row_power = 1.0 if self.kind == "offset_square" else 0.5

    @property
    def op_class(self):
        return "G" if self.kind == "softmax" else "P"

    def _heads(self, x, lin, dh):
        B, T, _ = x.shape
        return lin(x).view(B, T, self.cfg.heads, dh).transpose(1, 2)

    def forward(self, x, ctx):
        cfg = self.cfg
        cos, sin = ctx["rope"]
        q1 = apply_rope(self._heads(x, self.q1, cfg.d_key), cos, sin)
        k1 = apply_rope(self._heads(x, self.k1, cfg.d_key), cos, sin)
        v = self._heads(x, self.v, cfg.d_value)
        if self.kind == "softmax":
            if ctx["backend"] == "flex":
                y = softmax_attention_flex(q1, k1, v, ctx["doc_ids"])
            else:
                y = softmax_attention(q1, k1, v, ctx["visible"])
        else:
            q2 = k2 = None
            if hasattr(self, "q2"):
                q2 = apply_rope(self._heads(x, self.q2, cfg.d_key), cos, sin)
                k2 = apply_rope(self._heads(x, self.k2, cfg.d_key), cos, sin)
            if ctx["backend"] == "triton":  # scores/products in FP32 inside the kernel; inputs stay in compute dtype
                from .kernels.poly_attn import poly_attention_triton
                y = poly_attention_triton(self.kind, q1, k1, q2, k2, v, ctx["doc_ids"], ctx["row_scale"][self.row_power],
                                          coef=getattr(self, "coef", None), b=getattr(self, "b", None))
                B, H, T, dv = y.shape
                return self.o(y.transpose(1, 2).reshape(B, T, H * dv).to(x.dtype))
            if cfg.fp32_scores:  # at least FP32 (FP64 stays FP64 in reference tests)
                up = lambda t: None if t is None else t.to(torch.promote_types(t.dtype, torch.float32))
                q1, k1, q2, k2, v = map(up, (q1, k1, q2, k2, v))
            C = ctx["C"][self.row_power]
            with torch.autocast(x.device.type, enabled=False):
                y = polynomial_attention(self.kind, q1, k1, q2, k2, v, C,
                                         coef=getattr(self, "coef", None), b=getattr(self, "b", None))
            y = y.to(x.dtype)
        B, H, T, dv = y.shape
        return self.o(y.transpose(1, 2).reshape(B, T, H * dv))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, spectral: bool):
        super().__init__()
        self.kind = cfg.mlp
        bias = self.kind == "affine_bilinear"
        self.l = _linear(cfg.d_model, cfg.d_mlp, spectral, bias)
        self.r = _linear(cfg.d_model, cfg.d_mlp, spectral, bias)
        self.d = _linear(cfg.d_mlp, cfg.d_model, spectral, bias)

    @property
    def op_class(self):
        return "G" if self.kind == "swiglu" else "P"

    def forward(self, x):
        if self.kind == "swiglu":
            return self.d(F.silu(self.l(x)) * self.r(x))
        return self.d(self.l(x) * self.r(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        nin, nout, spectral = PACKAGES[cfg.normalization]
        self.alpha = cfg.residual_alpha
        if cfg.depth_residual_base:
            self.alpha *= (cfg.depth_residual_base / cfg.layers) ** cfg.depth_residual_exponent
        self.attn, self.mlp = Attention(cfg, spectral), MLP(cfg, spectral)
        mk = lambda k: make_norm(k, cfg.d_model, cfg) if k else nn.Identity()
        self.attn_in, self.attn_out = mk(nin), mk(nout)
        self.mlp_in, self.mlp_out = mk(nin), mk(nout)

        self.ablate: frozenset = frozenset()  # analysis only: {"attn", "mlp"} branches removed

    def forward(self, x, ctx):
        if self.ablate and self.training:
            raise RuntimeError("Block.ablate is analysis-only; reset it before training")
        if "attn" not in self.ablate:
            x = x + self.alpha * self.attn_out(self.attn(self.attn_in(x), ctx))
        if "mlp" not in self.ablate:
            x = x + self.alpha * self.mlp_out(self.mlp(self.mlp_in(x)))
        return x


class TensorLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.padded_vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = make_norm(cfg.final_normalization, cfg.d_model, cfg)
        self.unembed = nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False)
        if cfg.tie_embeddings:
            self.unembed.weight = self.embed.weight
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Brief §5 fallback init: embed N(0,1); Q/K/V/L/R fan_in^-1/2; D/O (2*layers*fan_in)^-1/2; unembed 0.02."""
        cfg = self.cfg
        nn.init.normal_(self.embed.weight, std=1.0)
        for blk in self.blocks:
            a, m = blk.attn, blk.mlp
            for lin in [a.q1, a.k1, a.v, m.l, m.r] + ([a.q2, a.k2] if hasattr(a, "q2") else []):
                nn.init.normal_(lin.weight, std=lin.weight.shape[1] ** -0.5)
            depth = cfg.depth_residual_base or cfg.layers  # never both depth conventions
            for lin in (a.o, m.d):
                nn.init.normal_(lin.weight, std=(2 * depth * lin.weight.shape[1]) ** -0.5)
            for lin in blk.modules():
                if isinstance(lin, (nn.Linear, SpectralLinear)) and lin.bias is not None:
                    lin.bias.zero_()
                if isinstance(lin, SpectralLinear):
                    lin.reset_gamma()
        if not cfg.tie_embeddings:
            nn.init.normal_(self.unembed.weight, std=0.02)

    @property
    def logit_multiplier(self) -> float:
        c = self.cfg
        return (c.width_scaling_base / c.d_model) ** c.width_logit_exponent if c.width_scaling_base else 1.0

    # ---- structure reporting -------------------------------------------------
    def operator_audit(self) -> dict:
        classes = {}
        for name, mod in self.named_modules():
            c = getattr(mod, "op_class", None)
            if isinstance(c, str):
                classes[name or "<root>"] = c
        overall = max(classes.values(), key=CLASS_ORDER.get) if classes else "P"
        return {"class": overall, "nodes": classes}

    def param_counts(self) -> dict:
        matrices = sum(p.numel() for n, p in self.named_parameters() if p.dim() >= 2)
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel() + (0 if self.cfg.tie_embeddings else self.unembed.weight.numel())
        return {"total": total, "matrices": matrices, "non_embedding_matrices": matrices - emb,
                "small": total - matrices}

    # ---- forward ---------------------------------------------------------------
    def make_ctx(self, idx, doc_ids):
        B, T = idx.shape
        if doc_ids is None:
            doc_ids = torch.zeros(B, T, dtype=torch.long, device=idx.device)
        pos = doc_positions(doc_ids)
        ctx = {"rope": rope_tables(pos, self.cfg.d_key, self.cfg.rope_base), "doc_ids": doc_ids}
        backend = self.cfg.attention_backend
        if backend == "auto":
            backend = "triton" if idx.is_cuda else "reference"
        if self.cfg.attention == "softmax":  # control: FlexAttention on CUDA (fast path), dense SDPA reference
            backend = "flex" if backend == "triton" else "softmax"
        ctx["backend"] = backend
        if ctx["backend"] == "triton":
            # n_visible = doc position + 1 for contiguous documents; padding rows get scale 0
            n = (pos + 1).float()
            valid = doc_ids >= 0
            ctx["row_scale"] = {p: torch.where(valid, n.pow(-p), torch.zeros_like(n)) for p in (0.5, 1.0)}
        elif ctx["backend"] != "flex":  # dense reference paths only; fast paths never build a T x T mask
            vis = visible_mask(doc_ids)
            ctx["visible"] = vis
            if self.cfg.attention != "softmax":
                ctx["C"] = {p: row_scale_mask(vis, p) for p in (0.5, 1.0)}
        return ctx

    def forward(self, idx, doc_ids=None, targets=None, reduction="mean"):
        ctx = self.make_ctx(idx, doc_ids)
        x = self.embed(idx)
        ckpt = self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x, ctx, use_reentrant=False) if ckpt else blk(x, ctx)
        x = self.final_norm(x)
        V = self.cfg.vocab_size  # padded vocab rows are never scored
        if targets is not None and self.cfg.loss_chunk_tokens > 0:
            from .kernels.fused_ce import linear_cross_entropy_sum
            logits = None  # full logits are never materialized
            w = self.unembed.weight[:V]
            if self.cfg.width_scaling_base:
                w = w * self.logit_multiplier
            loss = linear_cross_entropy_sum(x.flatten(0, 1), w, targets.flatten(), self.cfg.loss_chunk_tokens)
        else:
            logits = self.unembed(x)[..., :V]
            if self.cfg.width_scaling_base:
                logits = logits * self.logit_multiplier
            if targets is None:
                return logits
            loss = F.cross_entropy(logits.to(torch.promote_types(logits.dtype, torch.float32)).flatten(0, 1),
                                   targets.flatten(), ignore_index=-100,
                                   reduction="sum")
        n = (targets != -100).sum()
        if reduction == "sum":
            return logits, loss, n
        return logits, loss / n.clamp_min(1)
