"""Config dataclasses mirroring the contract in tensor_network_research_brief.md §11."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any

import yaml

MLP_KINDS = ("bilinear", "affine_bilinear", "swiglu")
ATTN_KINDS = ("signed_product", "offset_square", "lower_degree_mix", "softmax")
NORM_PACKAGES = ("pre_rms", "branch_rms", "sandwich_rms", "group_rms", "spectral_rms", "lagged", "taper")
FINAL_NORMS = ("rms", "lagged", "taper")


@dataclass
class ModelConfig:
    layers: int = 12
    d_model: int = 512
    heads: int = 8
    d_key: int = 64
    d_value: int = 64
    d_mlp: int = 1664
    vocab_size: int = 50257
    padded_vocab: int = 50304
    tie_embeddings: bool = False
    mlp: str = "bilinear"
    attention: str = "signed_product"
    normalization: str = "pre_rms"
    final_normalization: str = "rms"
    rms_eps: float = 1.0e-6
    norm_groups: int = 4
    qk_norm: bool = False
    dropout: float = 0.0
    position_encoding: str = "rope"
    rope_base: float = 10000.0
    initialization: str = "explicit_fan_in_with_depth_scaled_outputs"
    residual_alpha: float = 1.0
    # Multiplier on the Q/K init std (1.0 = brief §5 init). Small values start polynomial attention near zero, so huge-score
    # token pairs can't dominate input-independent normalizers at init (see docs/scaling.md "Polynomial normalization canaries").
    qk_init_scale: float = 1.0
    # Polynomial attention row scale n_visible^-p (None = brief §6 default: 0.5 signed, 1.0 offset_square). p = 1 is the visible-position
    # *average*: bounded even when many keys score coherently (e.g. a context of repeated tokens), where p = 0.5 still grows like sqrt(n).
    # Position-only, so the model stays exactly polynomial.
    attn_row_power: float | None = None
    # lagged scaling (only used by the lagged package / lagged final norm)
    lagged_half_life_tokens: float = 1.0e6
    # TaperNorm (package / final norm "taper"; arXiv 2602.10408): EMA rate of the warmup calibration statistics
    taper_ema_rate: float = 0.01
    # offset-square initial offset b
    offset_square_b: float = 1.0
    # fp32 score computation in the reference attention
    fp32_scores: bool = True
    # polynomial attention implementation: "reference" (dense, authoritative), "triton" (tiled fwd+bwd
    # kernel, CUDA only), or "auto" (triton on CUDA, reference elsewhere). For the softmax control the fast
    # path is FlexAttention with a causal+document block mask; "reference" is dense-mask SDPA.
    attention_backend: str = "auto"
    # recompute each block in backward (memory for compute); incompatible with lagged collection
    activation_checkpointing: bool = False
    # chunked fused unembedding + cross-entropy (no full logits) when targets are given; 0 = off
    loss_chunk_tokens: int = 0
    # Empirical width scaling (AdamW), NOT a muP derivation. None = explicit conventional parameterization
    # (default). With base width b and r = b / d_model: hidden-matrix LR *= r**width_lr_exponent and
    # logits *= r**width_logit_exponent; init unchanged (fan_in^-1/2). Exponents 0.5 flattened the
    # tnlm.coordcheck per-step-change slopes for this family (docs/scaling.md); textbook muP (1.0) overcorrected.
    width_scaling_base: int | None = None
    width_lr_exponent: float = 0.5
    width_logit_exponent: float = 0.5
    # Empirical depth scaling (AdamW): with base depth Lb, hidden-matrix LR *= (Lb / layers)**depth_lr_exponent.
    # Motivation: tnlm.coordcheck.run_depth shows the final residual's per-step change grows ~L^0.6 under the
    # default (docs/scaling.md). Not a CompleteP derivation.
    depth_scaling_base: int | None = None
    depth_lr_exponent: float = 0.5
    # Per-branch residual depth scaling (CompleteP-style experiment): with base depth Lb, every branch output is
    # multiplied by residual_alpha * (Lb / layers)**depth_residual_exponent, and the D/O init uses Lb instead of
    # `layers`, so the two depth conventions are never applied together. Identical to the default at layers == Lb.
    depth_residual_base: int | None = None
    depth_residual_exponent: float = 1.0

    def __post_init__(self):
        assert self.mlp in MLP_KINDS, self.mlp
        assert self.attention in ATTN_KINDS, self.attention
        assert self.normalization in NORM_PACKAGES, self.normalization
        assert self.final_normalization in FINAL_NORMS, self.final_normalization
        assert self.d_model % self.norm_groups == 0
        assert self.padded_vocab >= self.vocab_size
        assert self.dropout == 0.0, "dropout is not part of this model family"
        if self.normalization == "lagged" and self.final_normalization != "lagged":
            # Brief §3: a P package must freeze every activation norm, including the final one.
            raise ValueError("lagged package requires final_normalization: lagged (else label it N)")
        if self.normalization == "taper" and self.final_normalization != "taper":
            # All-Taper: only tapering the final norm too makes the frozen model exactly polynomial (arXiv 2602.10408).
            raise ValueError("taper package requires final_normalization: taper (else the final RMSNorm stays input-dependent)")
        assert self.attention_backend in ("reference", "triton", "auto"), self.attention_backend
        if self.activation_checkpointing and "taper" in (self.normalization, self.final_normalization):
            raise ValueError("activation checkpointing would double-count TaperNorm statistics and anchor terms")
        if self.activation_checkpointing and "lagged" in (self.normalization, self.final_normalization):
            raise ValueError("activation checkpointing would double-count lagged statistics")
        if self.qk_norm:
            raise NotImplementedError("qk_norm needs head scope/eps/gain/RoPE order spec (brief §11)")


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 4.5e-4
    beta1: float = 0.9
    beta2: float = 0.9872585449
    weight_decay_hidden: float = 0.1
    grad_clip: float = 1.0
    global_prediction_tokens_per_update: int = 8192
    rung_total_tokens: int = 100_000_000
    schedule: str = "constant_with_separate_cooldown"
    cooldown_fraction: float = 0.1
    bf16_compute: bool = True
    fp32_norm_reductions: bool = True
    fp32_optimizer_state: bool = True
    compile: bool = True
    seed: int = 101
    # CUDA bitwise determinism (torch.use_deterministic_algorithms + CUBLAS workspace), applied before model/NCCL init.
    # Verified bitwise-exact replay incl. resume at ~86% throughput. Brief §9: bitwise replay is required only for
    # deterministic tiny references; normal GPU restarts use stated tolerance, so throughput runs default to False.
    deterministic: bool = False
    # log tnlm.analysis.stability_stats on a fixed train probe row every N updates into metrics.jsonl (0 = off)
    stability_every_steps: int = 0
    # TaperNorm schedule (taper norms only): calibrate during warmup, freeze at warmup end, cosine-decay the shared gate
    # 1 -> 0 until taper_end_fraction of the run's optimizer steps; add taper_anchor_lambda * mean (r - s_tgt)^2 to the loss.
    taper_anchor_lambda: float = 0.1
    taper_end_fraction: float = 1.0
    # QK-clip (tnlm.qk_clip): every N optimizer steps, rescale any head whose probe max |S| exceeds tau (0 = off)
    qk_clip_tau: float = 0.0
    qk_clip_every_steps: int = 0
    # where QK-clip reads max |S|: "probe" (fixed probe row, exact) or "batch_bound" (every training batch, Cauchy-Schwarz upper bound;
    # tnlm.qk_clip.qk_clip_from_bounds_)
    qk_clip_source: str = "probe"
    # Muon (optimizer: muon): hidden matrices use torch.optim.Muon at `lr`; the rest AdamW at aux_adamw_lr
    muon_momentum: float = 0.95
    muon_adjust_lr: str = "match_rms_adamw"
    aux_adamw_lr: float | None = None


def _build(cls, d: dict[str, Any] | None):
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(model=_build(ModelConfig, raw.get("model")),
                   train=_build(TrainConfig, raw.get("train")), raw=raw)

    def to_dict(self) -> dict:
        """Full config: every raw section (project/data/evaluation/...) plus resolved model/train."""
        out = dict(self.raw)
        out.update(model=asdict(self.model), train=asdict(self.train))
        return out
