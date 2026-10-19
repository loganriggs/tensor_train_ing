"""Normalization modules (brief §3).

All reductions run in FP32. Each module reports its operator class:
  "N" = explicit input-dependent normalizer node, "P" = constant at a fixed checkpoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def upcast(x: torch.Tensor) -> torch.Tensor:
    """Reduction dtype: at least FP32 (keeps FP64 for reference tests)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


SIGMA_FLOOR = 1e-12


class DynamicNorm(nn.Module):
    """Input-dependent normalizer with analysis hooks on its denominator.

    denom_mode: None (live), "record" (live, store denominator), "clamp" (reuse the stored
    denominator; finite-ablation decomposition, brief §10.4) or "const" (use `denom_const`,
    a per-site fixed-denominator approximation, brief §10.5). Training always uses None.
    """
    op_class = "N"

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.denom_mode: str | None = None
        self.denom_saved: torch.Tensor | None = None
        self.denom_const: torch.Tensor | None = None

    def forward(self, x):
        if self.denom_mode is not None and self.training:
            raise RuntimeError("denom_mode is analysis-only; reset it before training")
        r = self.denom(x)
        if self.denom_mode == "record":
            self.denom_saved = r.detach()
        elif self.denom_mode == "clamp":
            r = self.denom_saved.to(r.dtype)
        elif self.denom_mode == "const":
            r = self.denom_const.to(device=r.device, dtype=r.dtype).expand_as(r)
        return (upcast(x) / r * upcast(self.weight)).to(x.dtype)


class RMSNorm(DynamicNorm):
    def denom(self, x: torch.Tensor) -> torch.Tensor:
        xf = upcast(x)
        return torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)


class GroupRMSNorm(DynamicNorm):
    """RMS over fixed contiguous feature groups (not token/batch groups)."""

    def __init__(self, d: int, groups: int = 4, eps: float = 1e-6):
        super().__init__(d, eps)
        assert d % groups == 0
        self.groups = groups

    def denom(self, x):
        xf = upcast(x).unflatten(-1, (self.groups, -1))
        r = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return r.expand_as(xf).flatten(-2)


class LaggedRMS(nn.Module):
    """Scalar running-RMS scaling with a lagged, detached statistic (brief §3, 'lagged').

    forward() divides by sqrt(previous_mean_square + eps); the current batch never
    influences its own output. While `collecting` is on, forward() also accumulates
    count-weighted sufficient statistics from x.detach() (padding excluded via
    `set_valid_mask`). `commit(tokens)` applies one EMA update and must be called
    exactly once per optimizer step by the training loop. Do not combine collection
    with activation recomputation (it would double count).
    """
    op_class = "P"

    def __init__(self, d: int, eps: float = 1e-6, half_life_tokens: float = 1e6):
        super().__init__()
        self.eps, self.half_life_tokens = eps, half_life_tokens
        self.weight = nn.Parameter(torch.ones(d))
        self.register_buffer("mean_sq", torch.ones((), dtype=torch.float64))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.register_buffer("acc_sum", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("acc_count", torch.zeros((), dtype=torch.float64), persistent=False)
        self.collecting = False
        self._valid: torch.Tensor | None = None

    def set_valid_mask(self, valid: torch.Tensor | None):
        self._valid = valid

    def denom(self, x):
        # sqrt in the FP64 statistic dtype first, then cast: mean_sq=1e60 -> divisor 1e30 fits FP32.
        r = torch.sqrt(self.mean_sq + self.eps)
        return r.to(torch.promote_types(x.dtype, torch.float32))

    def forward(self, x):
        if self.collecting:
            with torch.no_grad():
                sq = x.detach().double().pow(2).mean(-1)  # per token
                if self._valid is not None:
                    sq = sq * self._valid
                    n = self._valid.sum()
                else:
                    n = torch.tensor(float(sq.numel()), device=sq.device, dtype=sq.dtype)
                self.acc_sum += sq.sum()
                self.acc_count += n
        return (upcast(x) / self.denom(x) * upcast(self.weight)).to(x.dtype)

    @torch.no_grad()
    def commit(self, distributed: bool = True) -> float:
        """One EMA update per optimizer step using the accumulated token count.

        Under torch.distributed, statistics are summed across ranks first, so every data-parallel
        rank applies the identical update (buffers stay replicated)."""
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            stats = torch.stack([self.acc_sum, self.acc_count])
            torch.distributed.all_reduce(stats)
            self.acc_sum.copy_(stats[0]); self.acc_count.copy_(stats[1])
        n = float(self.acc_count)
        if n == 0:
            return float(self.mean_sq)
        batch_ms = self.acc_sum / self.acc_count
        if not bool(self.calibrated):
            self.mean_sq.copy_(batch_ms)
            self.calibrated.fill_(True)
        else:
            keep = 0.5 ** (n / self.half_life_tokens)
            self.mean_sq.mul_(keep).add_((1 - keep) * batch_ms)
        self.acc_sum.zero_()
        self.acc_count.zero_()
        if not torch.isfinite(self.mean_sq):
            raise FloatingPointError("non-finite lagged RMS statistic")
        return float(self.mean_sq)


class TaperRMS(nn.Module):
    """TaperNorm (arXiv 2602.10408), RMS form: g * x/r(x) * gamma + (1 - g) * c * x * gamma_tilde.

    g = 1 is ordinary RMSNorm. g = 0 is a fixed diagonal scale c * gamma_tilde that folds into the next linear map,
    so a model whose norms (final norm included, "All-Taper") are all frozen at g = 0 is exactly polynomial (class P).
    Lifecycle, driven by the training loop:
      1. warmup (g = 1, not frozen): with `collecting` on, forward() accumulates detached per-token statistics and
         `commit()` applies one EMA update (rate `ema_rate`) per optimizer step;
      2. `freeze()` at taper start: c = E[|x*gamma|^2 / r] / E[|x*gamma|^2] (the least-squares match of the linear
         branch to the normalized one), s_tgt = E[r] (bias-corrected EMAs), gamma_tilde <- gamma;
      3. a shared gate g decays 1 -> 0 (`taper_gate`); while frozen and training, forward() also accumulates the
         differentiable scale-anchor term (r(x) - s_tgt)^2, read and reset by `taper_anchor_loss`.
    Both gamma and gamma_tilde stay trainable. Don't combine with activation recomputation (double counting).
    """

    def __init__(self, d: int, eps: float = 1e-6, ema_rate: float = 0.01):
        super().__init__()
        self.eps, self.ema_rate = eps, ema_rate
        self.weight = nn.Parameter(torch.ones(d))
        self.weight_tilde = nn.Parameter(torch.ones(d))
        for name in ("gate", "c", "s_tgt", "ema_num", "ema_den", "ema_r", "ema_steps"):
            self.register_buffer(name, torch.tensor(1.0 if name in ("gate", "c", "s_tgt") else 0.0, dtype=torch.float64))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        for name in ("acc_num", "acc_den", "acc_r", "acc_count"):
            self.register_buffer(name, torch.zeros((), dtype=torch.float64), persistent=False)
        self.collecting = False
        self._valid: torch.Tensor | None = None
        self._anchor_sum: torch.Tensor | None = None
        self._anchor_count = 0.0

    @property
    def op_class(self):
        return "P" if bool(self.frozen) and float(self.gate) == 0.0 else "N"

    def set_valid_mask(self, valid: torch.Tensor | None):
        self._valid = valid

    def forward(self, x):
        xf = upcast(x)
        g = float(self.gate)
        need_r = g > 0 or self.collecting or (self.training and bool(self.frozen))
        r = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) if need_r else None
        if self.collecting and not bool(self.frozen):
            with torch.no_grad():
                sq = (xf.detach() * upcast(self.weight.detach())).double().pow(2).sum(-1)  # |x*gamma|^2 per token
                rr = r.detach().double().squeeze(-1)
                w = self._valid.double() if self._valid is not None else torch.ones_like(rr)
                self.acc_num += (sq / rr * w).sum(); self.acc_den += (sq * w).sum()
                self.acc_r += (rr * w).sum(); self.acc_count += w.sum()
        if self.training and bool(self.frozen):
            dev = (r.squeeze(-1) - self.s_tgt.to(r.dtype)).pow(2)
            if self._valid is not None:
                dev, n = dev * self._valid, float(self._valid.sum())
            else:
                n = float(dev.numel())
            s = dev.sum()
            self._anchor_sum = s if self._anchor_sum is None else self._anchor_sum + s
            self._anchor_count += n
        lin = float(self.c) * xf * upcast(self.weight_tilde)
        if g == 0.0:
            return lin.to(x.dtype)
        normed = xf / r * upcast(self.weight)
        return (normed if g == 1.0 else g * normed + (1 - g) * lin).to(x.dtype)

    @torch.no_grad()
    def commit(self, distributed: bool = True):
        """One bias-correctable EMA update of the warmup statistics; call once per optimizer step while collecting."""
        stats = torch.stack([self.acc_num, self.acc_den, self.acc_r, self.acc_count])
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        if float(stats[3]) > 0:
            mu = self.ema_rate
            for ema, v in ((self.ema_num, stats[0]), (self.ema_den, stats[1]), (self.ema_r, stats[2])):
                ema.mul_(1 - mu).add_(mu * v / stats[3])
            self.ema_steps += 1
        for a in (self.acc_num, self.acc_den, self.acc_r, self.acc_count):
            a.zero_()

    @torch.no_grad()
    def freeze(self):
        if bool(self.frozen):
            return
        n = float(self.ema_steps)
        if n == 0:
            raise RuntimeError("TaperRMS.freeze() before any committed warmup statistics")
        corr = 1 - (1 - self.ema_rate) ** n
        num, den, rr = self.ema_num / corr, self.ema_den / corr, self.ema_r / corr
        self.c.copy_(num / (den + 1e-30))
        self.s_tgt.copy_(rr)
        self.weight_tilde.copy_(self.weight)
        self.frozen.fill_(True)
        if not (torch.isfinite(self.c) and torch.isfinite(self.s_tgt)):
            raise FloatingPointError("non-finite TaperNorm calibration")


def taper_modules(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, TaperRMS)]


def taper_gate(step: int, start: int, end: int) -> float:
    """Shared cosine gate: 1 before `start`, 0 from `end` on (TaperNorm schedule)."""
    if step <= start:
        return 1.0
    if step >= end:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (step - start) / (end - start)))


@torch.no_grad()
def taper_set_gate_(model: nn.Module, g: float):
    for m in taper_modules(model):
        m.gate.fill_(g)


def taper_anchor_loss(model: nn.Module) -> torch.Tensor | None:
    """Mean over tapered norms of the token-mean (r - s_tgt)^2 accumulated since the last call; resets the accumulators."""
    terms = []
    for m in taper_modules(model):
        if m._anchor_sum is not None and m._anchor_count > 0:
            terms.append(m._anchor_sum / m._anchor_count)
        m._anchor_sum, m._anchor_count = None, 0.0
    return torch.stack(terms).mean() if terms else None


class SpectralLinear(nn.Module):
    """sigmaReparam: W_eff = gamma * W / sigma(W) with sigma = u^T W v from stored power-iteration vectors.

    forward() never mutates u/v (compile- and grad-accumulation-safe). The training loop calls
    `spectral_step(model)` exactly once per optimizer step, before the forward passes of that step.
    u/v are checkpointed buffers; exported effective matrices reproduce the checkpoint's logits.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.gamma = nn.Parameter(torch.ones(1))  # 1-D: FSDP2 cannot shard 0-D parameters
        self.register_buffer("u", F.normalize(torch.randn(d_out), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(d_in), dim=0))

    @torch.no_grad()
    def reset_gamma(self, iters: int = 50):
        """Call after init so W_eff starts equal to the initialized W."""
        if self.weight.is_meta:
            return
        for _ in range(iters):
            self._power_step()
        self.gamma.fill_(self.sigma_used().item())

    @torch.no_grad()
    def _power_step(self):
        """One power-iteration step. A vector that collapses to ~0 (e.g. W=0) keeps its previous value.

        u^T W v is an estimate (a lower bound) of the spectral norm, not a certified bound.
        """
        w = self.weight.detach()
        if hasattr(w, "full_tensor"):  # FSDP2 DTensor shard: power-iterate on the full matrix on every rank
            w = w.full_tensor()
        w = upcast(w)
        for vec, new in ((self.v, lambda: w.t() @ self.u.to(w.dtype)), (self.u, lambda: w @ self.v.to(w.dtype))):
            n = new()
            nn_ = n.norm()
            if torch.isfinite(nn_) and nn_ > SIGMA_FLOOR:
                vec.copy_(n / nn_)

    def _sigma(self):
        w = upcast(self.weight)
        return self.u.to(w.dtype) @ w @ self.v.to(w.dtype)

    def sigma_used(self):
        """Divisor actually used: |u^T W v| floored at SIGMA_FLOOR (W=0 -> W_eff=0, no division by zero)."""
        return self._sigma().abs().clamp_min(SIGMA_FLOOR)

    def effective_weight(self):
        return (self.gamma * self.weight / self.sigma_used()).to(self.weight.dtype)

    def forward(self, x):
        # (gamma / sigma) W x, applied to whichever is smaller (both exact; effective_weight() remains the export / analysis view):
        # scaling the output touches tokens*d_out elements, scaling the weight d_in*d_out. Measured fwd+bwd on the GPU (bf16): at a
        # training micro-batch (8192 tokens, 2048x6784) the weight path is 13% faster; at 512 tokens the output path is 40% faster.
        if x.numel() // x.shape[-1] > self.weight.shape[1]:
            return F.linear(x, self.effective_weight(), self.bias)
        y = F.linear(x, self.weight, None) * (self.gamma / self.sigma_used()).to(x.dtype)
        return y if self.bias is None else y + self.bias


def make_norm(kind: str, d: int, cfg) -> nn.Module:
    if kind == "rms":
        return RMSNorm(d, cfg.rms_eps)
    if kind == "group":
        return GroupRMSNorm(d, cfg.norm_groups, cfg.rms_eps)
    if kind == "lagged":
        return LaggedRMS(d, cfg.rms_eps, cfg.lagged_half_life_tokens)
    if kind == "taper":
        return TaperRMS(d, cfg.rms_eps, cfg.taper_ema_rate)
    raise ValueError(kind)


@torch.no_grad()
def spectral_step(model: nn.Module):
    """One power-iteration step on every SpectralLinear; call once per optimizer step."""
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            m._power_step()


@torch.no_grad()
def freeze_spectral(model: nn.Module) -> int:
    """Replace every SpectralLinear by a plain nn.Linear holding its effective weight (export path).
    Returns the number of replaced modules. Logits are unchanged by construction."""
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, SpectralLinear):
                w = child.effective_weight()
                plain = nn.Linear(w.shape[1], w.shape[0], bias=child.bias is not None, device=w.device, dtype=w.dtype)
                plain.weight.copy_(w)
                if child.bias is not None:
                    plain.bias.copy_(child.bias)
                setattr(parent, name, plain)
                count += 1
    return count


def dynamic_norms(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, DynamicNorm)]


def lagged_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LaggedRMS)]


def rms_jacobian(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Analytical Jacobian of RMSNorm at a single vector x (brief §3)."""
    d = x.numel()
    r = math.sqrt(float(x @ x) / d + eps)
    G = torch.diag(gamma)
    return G / r - torch.outer(gamma * x, x) / (d * r ** 3)
