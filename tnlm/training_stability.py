"""Fixed training-row diagnostics and clipping without consuming the training stream."""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from .data import PackedSampler, batch_hash
from .normalization import lagged_modules, taper_modules


def stability_interval(cfg):
    value = getattr(cfg.train, 'stability_every_steps', 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('train.stability_every_steps must be an integer >= 0')
    return value


def qk_clip_policy(cfg):
    tau = getattr(cfg.train, 'qk_clip_tau', 0.0)
    every = getattr(cfg.train, 'qk_clip_every_steps', 0)
    if type(tau) not in (int, float) or not math.isfinite(tau) or tau < 0:
        raise ValueError('train.qk_clip_tau must be finite and nonnegative')
    if type(every) is not int or every < 0 or (tau > 0 and every == 0):
        raise ValueError('train.qk_clip_every_steps must be a nonnegative integer, positive when clipping is enabled')
    if tau > 0 and cfg.model.normalization == 'spectral_rms':
        raise ValueError('QK clipping does not support spectral models')
    return float(tau), every


def _finite_json(value, path, nonfinite):
    if isinstance(value, dict):
        return {key: _finite_json(item, f'{path}.{key}', nonfinite) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, f'{path}[{index}]', nonfinite) for index, item in enumerate(value)]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            nonfinite.append(path)
            return None
        return float(value)
    return value


class TrainingStabilityProbe:
    """A fixed first row from an independent train-only sampler, capped at 512.

    Run on the eager model between optimizer steps. This preserves RNG and mode
    state, but does not copy weights or optimizer tensors. Sampling is read-only;
    clipping deliberately changes Q/K parameters, leaving optimizer state alone.
    CUDA timing includes completion of the probe.
    """
    def __init__(self, cfg, device):
        self.device = torch.device(device)
        data = cfg.raw['data']
        context = min(data.get('context', 512), 512)
        sampler = PackedSampler(data['manifest'], split='train', batch_size=1,
                                context=context, seed=cfg.train.seed)
        batch = sampler.next_batch()
        self.identity = {'batch_sha256': batch_hash(batch),
                         'manifest_sha256': sampler.manifest_hash, 'split': 'train',
                         'seed': cfg.train.seed, 'context': context, 'batch_size': 1}
        self.idx, self.doc = (tensor.to(self.device) for tensor in batch[:2])

    def _run(self, model, operation):
        started = time.monotonic()
        modes = [(module, module.training) for module in model.modules()]
        norms = [(module, module.collecting, module._valid) for module in lagged_modules(model) + taper_modules(model)]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == 'cuda' else [])
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                model.eval()
                for module, _, _ in norms:
                    module.collecting = False
                stats = operation(model, self.idx, self.doc)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
        finally:
            for module, training in modes:
                module.training = training
            for module, collecting, valid in norms:
                module.collecting, module._valid = collecting, valid
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        return stats, time.monotonic() - started

    def sample(self, model):
        from .analysis import stability_stats
        stats, seconds = self._run(model, stability_stats)
        nonfinite = []
        sanitized = _finite_json(stats, 'stats', nonfinite)
        return {'stats': sanitized, 'probe': dict(self.identity),
                'seconds': seconds, 'nonfinite_paths': nonfinite}

    def clip(self, model, tau):
        from .qk_clip import qk_clip_
        stats, seconds = self._run(model, lambda m, idx, doc: qk_clip_(m, idx, doc, tau))
        nonfinite = []
        _finite_json(stats, 'qk_clip', nonfinite)
        if nonfinite:
            raise FloatingPointError('nonfinite QK clipping result: ' + ', '.join(nonfinite))
        return {**stats, 'probe': dict(self.identity), 'seconds': seconds,
                'optimizer_state_policy': 'moments and step counters unchanged'}
