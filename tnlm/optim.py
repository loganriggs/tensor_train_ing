"""Optimizers (brief §5): AdamW reference and Muon (hidden matrices) + AdamW (everything else).

Muon is PyTorch's established `torch.optim.Muon` with its documented update scaling
(`adjust_lr_fn`, default "match_rms_adamw" so its LR is on an AdamW-like update-RMS scale).
It is applied to each logical hidden matrix separately: L, R, D, Q1/K1/Q2/K2/V/O are distinct
parameters, so any compute-side fusion of L and R must not concatenate them for the optimizer.
Embedding, unembedding, gains, biases and head scalars use AdamW.

Every param group carries `base_lr`; the trainer sets `lr = base_lr * schedule_multiplier`.
"""
from __future__ import annotations

import torch


def hidden_matrix(name: str, p: torch.Tensor) -> bool:
    """Linear-map weights inside blocks. Head scalars such as `coef` (H, 4) are 2-D but are not matrices."""
    return p.ndim == 2 and name.startswith("blocks.") and name.endswith(".weight")


def param_split(model):
    vocab = {id(model.embed.weight), id(model.unembed.weight)}
    hidden, vocab_p, small = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in vocab:
            vocab_p.append(p)
        elif hidden_matrix(name, p):
            hidden.append(p)
        else:
            small.append(p)
    return hidden, vocab_p, small


class MuonAdamW(torch.optim.Optimizer):
    """Muon on hidden matrices plus AdamW on the rest, behaving as one optimizer for schedulers
    and checkpoints (`param_groups`, `state_dict`, `load_state_dict`)."""

    def __init__(self, model, *, lr, aux_lr, weight_decay, betas, momentum=0.95, nesterov=True,
                 adjust_lr_fn="match_rms_adamw", vocab_weight_decay=0.0):
        hidden, vocab_p, small = param_split(model)
        self.muon = torch.optim.Muon([{"params": hidden, "base_lr": lr, "kind": "muon"}], lr=lr,
                                     weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                     adjust_lr_fn=adjust_lr_fn)
        groups = [{"params": vocab_p, "weight_decay": vocab_weight_decay, "base_lr": aux_lr, "kind": "adamw_vocab"},
                  {"params": small, "weight_decay": 0.0, "base_lr": aux_lr, "kind": "adamw_small"}]
        self.adamw = torch.optim.AdamW([g for g in groups if g["params"]], lr=aux_lr, betas=betas)
        # deliberately not calling Optimizer.__init__: expose the children's groups/state directly
        self.defaults = {}
        self._optimizer_step_pre_hooks, self._optimizer_step_post_hooks = {}, {}

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def state(self):
        return {**self.muon.state, **self.adamw.state}

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])


def make_optimizer(model, cfg):
    """cfg: TrainConfig. optimizer 'adamw' reproduces the reference grouping; 'muon' = MuonAdamW."""
    betas = (cfg.beta1, cfg.beta2)
    mcfg = getattr(model, "cfg", None)
    width_mult = ((mcfg.width_scaling_base / mcfg.d_model) ** mcfg.width_lr_exponent
                  if mcfg is not None and mcfg.width_scaling_base else 1.0)
    if mcfg is not None and mcfg.depth_scaling_base:
        width_mult *= (mcfg.depth_scaling_base / mcfg.layers) ** mcfg.depth_lr_exponent
    if cfg.optimizer == "adamw":
        hidden, vocab_p, small = param_split(model)
        groups = [{"params": hidden, "weight_decay": cfg.weight_decay_hidden / width_mult, "base_lr": cfg.lr * width_mult,
                   "lr": cfg.lr * width_mult},
                  {"params": vocab_p + small, "weight_decay": 0.0, "base_lr": cfg.lr}]
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=betas, fused=torch.cuda.is_available()
                                 and all(p.is_cuda for p in model.parameters()))
    if cfg.optimizer == "muon":
        if width_mult != 1.0:
            raise NotImplementedError("width/depth LR scaling is defined for AdamW only; Muon uses its own shape scaling")
        return MuonAdamW(model, lr=cfg.lr, aux_lr=cfg.aux_adamw_lr or cfg.lr, weight_decay=cfg.weight_decay_hidden,
                         betas=betas, momentum=cfg.muon_momentum, adjust_lr_fn=cfg.muon_adjust_lr)
    raise ValueError(cfg.optimizer)
