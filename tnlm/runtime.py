"""Shared configuration identity and random-seed helpers."""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch


def full_config(cfg):
    out = {**cfg.raw, **cfg.to_dict()}
    project = dict(out.get('project', {}))
    project['spend_ledger'] = str(Path(project.get('spend_ledger', 'runs/spend.json')).resolve())
    out['project'] = project
    if hasattr(cfg.train, 'deterministic'):
        out['train']['deterministic'] = deterministic_requested(cfg)
    if hasattr(cfg.train, 'stability_every_steps'):
        from .training_stability import stability_interval
        out['train']['stability_every_steps'] = stability_interval(cfg)
    if any(hasattr(cfg.train, name) for name in ('qk_clip_tau', 'qk_clip_every_steps')):
        from .training_stability import qk_clip_policy
        tau, every = qk_clip_policy(cfg)
        out['train'].update(qk_clip_tau=tau, qk_clip_every_steps=every)
    return out



def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _default_missing_config(saved, expected):
    """Resolve absent dataclass fields only when the requested value is the default.

    Never rewrite recorded values or infer defaults for raw data/project sections.
    The returned copy is for validation; checkpoint provenance remains original.
    """
    from dataclasses import MISSING, fields
    from .config import ModelConfig, TrainConfig
    resolved = copy.deepcopy(saved)
    defaulted = {}
    for section, schema in (('model', ModelConfig), ('train', TrainConfig)):
        values = resolved.get(section)
        if not isinstance(values, dict):
            raise ValueError(f'resume config mismatch: missing {section} section')
        defaults = {f.name: f.default if f.default is not MISSING else f.default_factory()
                    for f in fields(schema)
                    if f.default is not MISSING or f.default_factory is not MISSING}
        # Runtime support predates the schema addition; retain legacy false semantics.
        if section == 'train':
            defaults.setdefault('deterministic', False)
            defaults.setdefault('stability_every_steps', 0)
            defaults.setdefault('qk_clip_tau', 0.0)
            defaults.setdefault('qk_clip_every_steps', 0)
        for name, default in defaults.items():
            if name not in values and name in expected.get(section, {}):
                current = expected[section][name]
                if type(current) is not type(default) or current != default:
                    raise ValueError(f'resume config mismatch: absent {section}.{name} requires its dataclass default')
                values[name] = copy.deepcopy(default)
                defaulted[f'{section}.{name}'] = copy.deepcopy(default)
    return resolved, {'rule': 'missing dataclass field requires current default',
                      'defaulted_fields': defaulted}


def resume_validator(cfg, *, context_fork=False):
    """Allow execution settings and upward token promotion, freezing model/data/math."""
    if type(context_fork) is not bool:
        raise ValueError('context_fork must be bool')
    expected = full_config(cfg)
    def validate(saved):
        right = copy.deepcopy(expected)
        left, compatibility = _default_missing_config(saved, right)
        if context_fork:
            old_context = left.get('data', {}).get('context', 512)
            new_context = right.get('data', {}).get('context', 512)
            if (type(old_context) is not int or type(new_context) is not int
                    or old_context <= 0 or new_context <= old_context):
                raise ValueError('context fork requires a strictly increasing positive integer context')
            left.setdefault('data', {})['context'] = new_context
            right.setdefault('data', {})['context'] = new_context
        previous_project = left.pop('project', {})
        current_project = right.pop('project', {})
        previous_ledger = Path(previous_project.get('spend_ledger', 'runs/spend.json')).resolve()
        current_ledger = Path(current_project.get('spend_ledger', 'runs/spend.json')).resolve()
        if previous_ledger != current_ledger:
            raise ValueError('resume must retain the project spend ledger')
        left['train'].setdefault('deterministic', False)
        right['train'].setdefault('deterministic', False)
        previous = left['train'].pop('rung_total_tokens')
        requested = right['train'].pop('rung_total_tokens')
        if (left['train'].get('schedule') == 'cosine' or right['train'].get('schedule') == 'cosine') and requested != previous:
            raise ValueError('cosine resume must retain the exact original token horizon')
        if requested != previous and any(
                side.get('model', {}).get(field) == 'taper'
                for side in (left, right) for field in ('normalization', 'final_normalization')):
            raise ValueError('TaperNorm resume must retain the exact original token horizon')
        if requested < previous or left != right:
            raise ValueError('resume config mismatch: only upward token promotion and project execution settings may change')
        return compatibility
    return validate


def record_context_fork(progress, *, old_context, new_context, parent, source_state):
    """Bind a context intervention to the restored global source watermark."""
    history = copy.deepcopy(progress.get('context_history'))
    if history is None:
        history = [{'context': old_context, 'start_tokens': None, 'start_step': None,
                    'source': 'legacy_unknown'}]
    history.append({'context': new_context, 'start_tokens': progress['tokens'],
                    'start_step': progress['steps'], 'source': 'context_fork',
                    'parent': copy.deepcopy(parent), 'source_sampler': copy.deepcopy(source_state)})
    progress['context_history'] = history
    progress.setdefault('context_fork_exposure', {'prefix_tokens_at_or_before_watermark': 0,
                                               'masked_prediction_targets': 0})



def evaluation_validator(cfg):
    """Freeze mathematical/data/probe identity while allowing execution billing changes."""
    compatibility = resume_validator(cfg)
    expected_train = full_config(cfg)['train']
    expected_train.setdefault('deterministic', False)
    def validate(saved):
        metadata = compatibility(saved)
        normalized, _ = _default_missing_config(saved, full_config(cfg))
        saved_train = normalized['train']
        saved_train.setdefault('deterministic', False)
        if saved_train != expected_train:
            raise ValueError('evaluation config mismatch: checkpoint training identity is pinned')
        return metadata
    return validate



def deterministic_requested(cfg):
    value = getattr(cfg.train, 'deterministic', False)
    if type(value) is not bool:
        raise ValueError('train.deterministic must be Boolean')
    return value


def determinism_metadata():
    import os
    return {'algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'scope': 'PyTorch deterministic policy; custom kernels and cross-platform replay require independent verification'}


def training_execution(*, collective=False, default_device='cuda'):
    """Scope process-wide numerical policy to one synchronous training call.

    Concurrent training calls in one process are unsupported. Dry planning does
    not mutate global settings. Collective setup failures propagate on all ranks.
    """
    from functools import wraps
    from contextlib import ExitStack
    def decorate(function):
        @wraps(function)
        def wrapped(cfg, *args, **kwargs):
            from .config import Config
            if collective:
                function.__globals__['_group']()
                phase = function.__globals__['_phase']
            else:
                phase = lambda _name, call: call()
            cfg = phase('determinism configuration load', lambda: Config.from_yaml(str(cfg))
                        if isinstance(cfg, (str, Path)) else cfg)
            enabled = phase('determinism validation', lambda: deterministic_requested(cfg))
            if cfg.raw.get('project', {}).get('dry_run', True):
                return function(cfg, *args, **kwargs)
            with ExitStack() as stack:
                phase('determinism setup', lambda: stack.enter_context(
                    deterministic_execution(enabled, device=kwargs.get('device', default_device))))
                return function(cfg, *args, **kwargs)
        return wrapped
    return decorate


from contextlib import contextmanager


@contextmanager
def deterministic_execution(enabled, *, device):
    """Scope algorithm flags; retain cuBLAS configuration once CUDA is initialized.

    cuBLAS may cache handles, so changing its environment back after initialization
    would misrepresent the process configuration. CPU-only calls restore it.
    """
    import os
    if type(enabled) is not bool:
        raise ValueError('train.deterministic must be Boolean')
    previous = determinism_metadata()
    if enabled and torch.device(device).type == 'cuda' and torch.cuda.is_initialized():
        if previous['cublas_workspace_config'] not in (':4096:8', ':16:8'):
            raise RuntimeError('deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                               'before CUDA initialization; launch a fresh process')
    try:
        if enabled:
            if previous['cublas_workspace_config'] not in (':4096:8', ':16:8'):
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(enabled, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous['algorithms_enabled'],
                                           warn_only=previous['warn_only'])
        torch.backends.cudnn.benchmark = previous['cudnn_benchmark']
        torch.backends.cudnn.deterministic = previous['cudnn_deterministic']
        if enabled and torch.device(device).type == 'cuda' and torch.cuda.is_initialized():
            pass  # Keep the configuration under which CUDA handles were created.
        elif previous['cublas_workspace_config'] is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = previous['cublas_workspace_config']
