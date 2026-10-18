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


def qk_clip_source(cfg):
    source = getattr(cfg.train, 'qk_clip_source', 'probe')
    if type(source) is not str or source not in ('probe', 'batch_bound'):
        raise ValueError('train.qk_clip_source must be probe or batch_bound')
    if source == 'batch_bound' and cfg.model.attention == 'softmax':
        raise ValueError('batch_bound recording requires polynomial attention')
    return source


def qk_clip_policy(cfg):
    qk_clip_source(cfg)
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


class BatchScoreBounds:
    """Record pre-update forward bounds; clip from the final original microbatch.

    The metric spans every executed microbatch, layer and factor. It includes
    padding and unscored forward positions and is not a post-update score cap.
    Detached snapshots prevent backward recomputation from changing the source.
    """
    def __init__(self, model, cfg):
        from .qk_clip import set_score_bound_recording
        self.model = model
        self.enabled = qk_clip_source(cfg) == 'batch_bound'
        self.maximum = None
        self.last = None
        if self.enabled:
            set_score_bound_recording(model, True)

    def begin_update(self):
        self.maximum = self.last = None

    def observe(self):
        if not self.enabled:
            return
        from .qk_clip import score_bounds
        snapshot = score_bounds(self.model)
        entries = []
        for layer, (block, bounds) in enumerate(zip(self.model.blocks, snapshot)):
            expected = {'S1', 'S2'} if hasattr(block.attn, 'q2') else {'S1'}
            if set(bounds) != expected:
                raise ValueError('missing or invalid polynomial score bounds')
            for name, bound in bounds.items():
                if not torch.is_tensor(bound) or bound.shape != (self.model.cfg.heads,):
                    raise ValueError('invalid score bound shape')
                entries.append((layer, name, bound.detach()))
        if not entries:
            raise ValueError('no polynomial score bounds were recorded')
        # One compact copy protects against reused compiled output buffers and
        # allows one reduction instead of launching kernels for every factor.
        values = torch.cat([bound for _, _, bound in entries])
        maximum = values.amax()
        maximum = torch.where((values < 0).any(), maximum.new_full((), float('nan')), maximum)
        self.maximum = maximum if self.maximum is None else torch.maximum(self.maximum, maximum)
        self.last = [{} for _ in snapshot]
        heads = self.model.cfg.heads
        for offset, (layer, name, _) in enumerate(entries):
            self.last[layer][name] = values[offset * heads:(offset + 1) * heads]

    def finish_update(self):
        if not self.enabled:
            return 0.0
        if self.maximum is None:
            raise ValueError('no score bounds recorded for this update')
        maximum = float(self.maximum)
        if not math.isfinite(maximum) or maximum < 0:
            raise FloatingPointError('nonfinite or negative score bound; update aborted')
        return maximum

    def clip(self, tau):
        from .qk_clip import qk_clip_from_bounds_
        self.finish_update()
        if self.last is None:
            raise ValueError('no last-microbatch score bounds for clipping')
        for block, bounds in zip(self.model.blocks, self.last):
            block.attn.last_score_bound = bounds
        started = time.monotonic()
        stats = qk_clip_from_bounds_(self.model, tau)
        nonfinite = []
        _finite_json(stats, 'qk_clip', nonfinite)
        if nonfinite:
            raise FloatingPointError('nonfinite QK clipping result: ' + ', '.join(nonfinite))
        return {**stats, 'seconds': time.monotonic() - started,
                'optimizer_state_policy': 'moments and step counters unchanged'}


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
        bounds = [(block.attn, block.attn.record_score_bound, block.attn.last_score_bound)
                  for block in model.blocks]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == 'cuda' else [])
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                model.eval()
                for attention, _, _ in bounds:
                    attention.record_score_bound = False
                for module, _, _ in norms:
                    module.collecting = False
                stats = operation(model, self.idx, self.doc)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
        finally:
            for attention, recording, last in bounds:
                attention.record_score_bound, attention.last_score_bound = recording, last
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
