"""Shared, cumulative-step TaperNorm lifecycle for the training loops."""
from __future__ import annotations

import math

import torch

from .normalization import taper_anchor_loss, taper_gate, taper_modules, taper_set_gate_


def is_taper_config(cfg):
    return 'taper' in (cfg.model.normalization, cfg.model.final_normalization)


def _number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    return float(value)


def validate_taper_config(cfg):
    """Validate enabled Taper policy without constructing a model or output directory."""
    if not is_taper_config(cfg):
        return False
    anchor = _number(cfg.train.taper_anchor_lambda, 'train.taper_anchor_lambda')
    end = _number(cfg.train.taper_end_fraction, 'train.taper_end_fraction')
    ema = _number(cfg.model.taper_ema_rate, 'model.taper_ema_rate')
    if anchor < 0:
        raise ValueError('train.taper_anchor_lambda must be nonnegative')
    if not 0 < end <= 1:
        raise ValueError('train.taper_end_fraction must be in (0, 1]')
    if not 0 < ema <= 1:
        raise ValueError('model.taper_ema_rate must be in (0, 1]')
    if cfg.train.compile:
        raise ValueError('TaperNorm training requires train.compile: false')
    if cfg.model.activation_checkpointing:
        raise ValueError('TaperNorm cannot use activation checkpointing')
    return True


class TaperTraining:
    """Commit warmup statistics once per update and save the gate at completed steps.

    During warmup callers supply a nonpadding mask. Once frozen they supply the
    selected prediction-target mask, so anchor means can be weighted like CE.
    Clearing the mask also disables collection for probes and checkpoint boundaries.
    """
    def __init__(self, model, cfg, *, total_steps, warmup_steps, distributed=False):
        self.model = model
        self.enabled = validate_taper_config(cfg)
        self.norms = taper_modules(model) if self.enabled else []
        self.anchor_lambda = float(cfg.train.taper_anchor_lambda) if self.enabled else 0.0
        self.distributed = distributed
        self.warmup_steps = warmup_steps
        self.end_step = total_steps
        self._step = None
        self._gate_used = 1.0
        self._constants_logged = False
        if self.enabled:
            if type(total_steps) is not int or total_steps <= 0:
                raise ValueError('TaperNorm requires positive integer total_steps')
            self.end_step = round(cfg.train.taper_end_fraction * total_steps)
            if type(warmup_steps) is not int or warmup_steps <= 0 or self.end_step <= warmup_steps:
                raise ValueError('TaperNorm endpoint must be strictly after positive warmup_steps')
            if not self.norms:
                raise ValueError('TaperNorm config has no TaperRMS modules')

    @property
    def collecting(self):
        return self.enabled and self._step is not None and self._step < self.warmup_steps

    def _freeze(self):
        for norm in self.norms:
            norm.collecting = False
            norm.freeze()

    def before_step(self, k):
        if not self.enabled:
            return
        if type(k) is not int or k < 0:
            raise ValueError('TaperNorm step must be a nonnegative integer')
        self._step = k
        if k >= self.warmup_steps:
            self._freeze()
        elif any(bool(norm.frozen) for norm in self.norms):
            raise ValueError('frozen TaperNorm cannot resume inside warmup')
        self._gate_used = taper_gate(k, self.warmup_steps, self.end_step)
        taper_set_gate_(self.model, self._gate_used)
        for norm in self.norms:
            norm.collecting = self.collecting

    def set_valid_mask(self, mask):
        for norm in self.norms:
            norm.set_valid_mask(mask)
            norm.collecting = mask is not None and self.collecting

    def anchor(self):
        return taper_anchor_loss(self.model) if self.enabled else None

    def after_step(self, completed_steps):
        if not self.enabled:
            return
        if self._step is None or completed_steps != self._step + 1:
            raise ValueError('TaperNorm after_step must follow exactly one optimizer update')
        collect = self.collecting
        self.set_valid_mask(None)
        if collect:
            # All ranks must complete every module's collective before a local
            # finite-statistic failure can escape to the trainer phase barrier.
            for norm in self.norms:
                norm.commit(distributed=self.distributed)
            for norm in self.norms:
                if not all(bool(torch.isfinite(getattr(norm, key)))
                           for key in ('ema_num', 'ema_den', 'ema_r', 'ema_steps')):
                    raise FloatingPointError('nonfinite TaperNorm warmup statistics')
        self.set_valid_mask(None)
        if completed_steps >= self.warmup_steps:
            self._freeze()
        taper_set_gate_(self.model, taper_gate(completed_steps, self.warmup_steps, self.end_step))
        self._step = None

    def metrics(self, anchor_average):
        if not self.enabled:
            return {}
        result = {'taper_gate': float(self.norms[0].gate), 'taper_gate_used': self._gate_used,
                  'taper_anchor': float(anchor_average)}
        if not self._constants_logged and all(bool(norm.frozen) for norm in self.norms):
            ids = {id(norm) for norm in self.norms}
            result['taper_constants'] = {
                name: {'c': float(norm.c), 's_tgt': float(norm.s_tgt)}
                for name, norm in self.model.named_modules() if id(norm) in ids}
            self._constants_logged = True
        return result
