"""Scheduled optimizer-group exposure, without executing or changing an optimizer.

Both AdamW and PyTorch Muon decay only parameters with non-None gradients.
Products below therefore describe a parameter participating in every observed
update, not an assertion about every group member's actual history. Muon's decay
uses raw group LR; its Newton-Schulz update uses a separate shape-adjusted LR.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import torch

from .optim import MuonAdamW


def _scalar(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f'{name} must be scalar')
        value = value.item()
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def update_optimizer_exposure(progress, optimizer) -> dict:
    """Record one successful real update using pre-update progress counters.

    Call AFTER optimizer.step and BEFORE incrementing progress['steps']/['tokens'].
    Validates all inputs before replacing progress['optimizer_exposure']; resume
    preserves prior totals. Missing history at a nonzero start is explicitly
    incomplete. Ordered shape/dtype group identities detect membership-layout
    changes; without model names they cannot distinguish equal-shaped swaps.
    """
    if type(optimizer) is torch.optim.AdamW:
        children = [('adamw', optimizer)]
    elif (type(optimizer) is MuonAdamW and type(optimizer.muon) is torch.optim.Muon
          and type(optimizer.adamw) is torch.optim.AdamW):
        children = [('muon', optimizer.muon), ('adamw', optimizer.adamw)]
    else:
        raise ValueError('exposure supports exactly AdamW and MuonAdamW')
    steps, tokens = progress.get('steps', 0), progress.get('tokens', 0)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (steps, tokens)):
        raise ValueError('progress steps/tokens must be nonnegative integers')
    groups = []
    for kind, child in children:
        for group in child.param_groups:
            identity = {'index': len(groups), 'optimizer_kind': kind,
                        'kind': group.get('kind'),
                        'parameters': [{'shape': list(p.shape), 'dtype': str(p.dtype)} for p in group['params']],
                        'parameter_names': copy.deepcopy(group.get('param_names'))}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            lr = _scalar(group['lr'], 'lr')
            decay = _scalar(group.get('weight_decay', 0.), 'weight_decay')
            product = lr * decay
            if not math.isfinite(product):
                raise ValueError('lr*weight_decay must be finite')
            groups.append((identity, digest, lr, product))
    previous = progress.get('optimizer_exposure')
    if previous is None:
        exposure = {'version': 1, 'complete_history': steps == 0 and tokens == 0,
                    'known_start_steps': steps, 'known_start_tokens': tokens,
                    'observed_steps': 0,
                    'scope': 'scheduled group exposure; mathematical shrinkage product before floating-point rounding, for a parameter participating in every observed update',
                    'muon_decay_semantics': 'raw group lr; Newton-Schulz update separately shape-adjusted',
                    'groups': []}
        for identity, digest, _, _ in groups:
            exposure['groups'].append({'identity': identity, 'identity_sha256': digest,
                'observed_steps': 0, 'sum_lr': 0., 'sum_lr_weight_decay': 0.,
                'sum_log_abs_nonzero_factors': 0., 'zero_factors': 0, 'negative_factors': 0,
                'shrinkage_sign': 1, 'shrinkage_log_abs': 0.})
    else:
        exposure = copy.deepcopy(previous)
        if exposure.get('version') != 1 or len(exposure['groups']) != len(groups):
            raise ValueError('optimizer exposure schema/group mismatch')
        if exposure['known_start_steps'] + exposure['observed_steps'] != steps:
            raise ValueError('optimizer exposure step history mismatch')
    for record, (identity, digest, lr, product) in zip(exposure['groups'], groups):
        if record['identity_sha256'] != digest or record['identity'] != identity:
            raise ValueError('optimizer exposure group identity mismatch')
        record['sum_lr'] += lr
        record['sum_lr_weight_decay'] += product
        record['observed_steps'] += 1
        factor = 1. - product
        if factor == 0:
            record['zero_factors'] += 1
        else:
            record['sum_log_abs_nonzero_factors'] += (math.log1p(-product) if factor > 0 else math.log(-factor))
            record['negative_factors'] += int(factor < 0)
        record['shrinkage_sign'] = (0 if record['zero_factors'] else (-1 if record['negative_factors'] % 2 else 1))
        record['shrinkage_log_abs'] = (None if record['zero_factors'] else record['sum_log_abs_nonzero_factors'])
    exposure['observed_steps'] += 1
    # Overflow or corrupted previous totals must fail before mutating progress.
    try:
        json.dumps(exposure, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('optimizer exposure must remain finite JSON') from exc
    progress['optimizer_exposure'] = exposure
    return exposure
