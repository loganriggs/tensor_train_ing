"""Meta-only checkpoint retention projection; no reservation or deletion."""
from __future__ import annotations

import math
import bisect
import heapq

import torch

from .model import TensorLM
from .optim import hidden_matrix

_METADATA_ALLOWANCE = 1024 * 1024
_SERIALIZATION_OVERHEAD = 1024 * 1024
_EARLY = (1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def checkpoint_retention_policy(cfg) -> dict:
    """Resolve full-state retention, including opt-in disposable screening runs.

    Zero recoveries disables recovery saves altogether. Initial saves can be
    omitted independently; final, milestone, and branch checkpoints are retained.
    """
    project = cfg.raw.get('project', {})
    keep = _integer(project.get('keep_recovery_checkpoints', 2),
                    'project.keep_recovery_checkpoints')
    initial = project.get('save_initial_checkpoint', True)
    if not isinstance(initial, bool):
        raise ValueError('project.save_initial_checkpoint must be bool')
    return {'keep_recovery_checkpoints': keep, 'save_initial_checkpoint': initial}


def milestone_policy(cfg) -> dict:
    """Resolve ordinary milestones; an explicit list replaces the entire default."""
    project = cfg.raw.get('project', {})
    if 'milestone_tokens' not in project:
        return {'mode': 'default', 'tokens': None}
    values = project['milestone_tokens']
    if not isinstance(values, list):
        raise ValueError('project.milestone_tokens must be a list of positive integers')
    for value in values:
        _integer(value, 'project.milestone_tokens entry', 1)
    return {'mode': 'explicit', 'tokens': sorted(set(values))}


def pending_milestone_tokens(cfg, *, start_tokens, token_limit, reachable_tokens=None,
                             include_branch=False):
    """Return lazy sorted unique thresholds in (start, min(limit, reachable)].

    Explicit configuration replaces all ordinary milestones. Optional mandatory
    90% branch is merged by value for the serial trainer; distributed trainers
    should keep their branchpoint separately. Default 100M ranges stay lazy.
    Validation is eager even when the returned iterator is never consumed.
    """
    start = _integer(start_tokens, 'start_tokens')
    limit = _integer(token_limit, 'token_limit')
    end = limit if reachable_tokens is None else min(limit, _integer(reachable_tokens, 'reachable_tokens'))
    if not isinstance(include_branch, bool):
        raise ValueError('include_branch must be bool')
    policy = milestone_policy(cfg)
    if policy['mode'] == 'explicit':
        values = policy['tokens']
        ordinary = iter(values[bisect.bisect_right(values, start):bisect.bisect_right(values, end)])
        sources = [ordinary]
    else:
        early = (value for value in _EARLY if start < value <= end)
        hundreds = range((start // 100_000_000 + 1) * 100_000_000, end + 1, 100_000_000)
        sources = [early, hundreds]
    branch = (9 * limit) // 10
    if include_branch and cfg.train.schedule != 'cosine' and start < branch <= end:
        sources.append((branch,))
    def unique():
        previous = None
        for value in heapq.merge(*sources):
            if value != previous:
                yield value
                previous = value
    return unique()


def checkpoint_storage_plan(cfg, *, start_tokens=0, token_limit=None, max_steps=None,
                            distributed_world_size=1, distributed=False,
                            export_weights=False, cooldown=False) -> dict:
    """Project mature checkpoint bytes and peak planned retained/staged count.

    Uses current default floating dtype. No real tensors, model initialization,
    RNG draws, optimizer construction, disk writes or free-space checks occur.
    Bounds are advisory estimates, not guaranteed upper bounds: existing artifacts,
    concurrent runs/pinned old recoveries, arbitrary metadata/code, and filesystem
    allocation/compression are excluded. Time caps can reduce reachable updates.
    """
    start = _integer(start_tokens, 'start_tokens')
    limit = _integer(cfg.train.rung_total_tokens if token_limit is None else token_limit, 'token_limit')
    world = _integer(distributed_world_size, 'distributed_world_size', 1)
    if not distributed and world != 1:
        raise ValueError('distributed_world_size must be 1 for a serial plan')
    for name, value in [('distributed', distributed), ('export_weights', export_weights), ('cooldown', cooldown)]:
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be bool')
    batch = _integer(cfg.train.global_prediction_tokens_per_update, 'global_prediction_tokens_per_update', 1)
    retention = checkpoint_retention_policy(cfg)
    keep = retention['keep_recovery_checkpoints']
    remaining = max(0, limit - start)
    updates = (remaining + batch - 1) // batch
    if max_steps is not None:
        updates = min(updates, _integer(max_steps, 'max_steps'))
    reachable = min(limit, start + updates * batch) if start < limit else start
    policy = milestone_policy(cfg)
    end = min(limit, reachable)
    if policy['mode'] == 'default':
        early = sum(start < value <= end for value in _EARLY)
        hundreds = max(0, end // 100_000_000 - start // 100_000_000)
        milestone_count = early + hundreds
    else:
        values = policy['tokens']
        milestone_count = max(0, bisect.bisect_right(values, end) - bisect.bisect_right(values, start))
    branch = (9 * limit) // 10
    branch_reachable = not cooldown and cfg.train.schedule != 'cosine' and start < branch <= end
    branch_is_milestone = (branch in policy['tokens'] if policy['mode'] == 'explicit' else
                          branch in _EARLY or (branch > 0 and branch % 100_000_000 == 0))
    separate_branch = int(branch_reachable and distributed)
    if branch_reachable and not distributed and not branch_is_milestone:
        milestone_count += 1
    rolling = min(updates, keep + 1) if keep else 0  # New recovery before pruning.

    class MetaModel(TensorLM):
        def reset_parameters(self):
            pass

    with torch.device('meta'):
        model = MetaModel(cfg.model)
    if cfg.train.optimizer not in ('adamw', 'muon'):
        raise ValueError('checkpoint plan supports adamw or muon')
    seen = set()
    parameter_bytes = 0
    optimizer_bytes = 0
    shard_padding = 0
    step_scalars = 0
    for name, parameter in model.named_parameters():
        storage = parameter.untyped_storage()
        if storage._cdata not in seen:
            seen.add(storage._cdata)
            parameter_bytes += storage.nbytes()
        if not parameter.requires_grad:
            continue
        moments = 1 if cfg.train.optimizer == 'muon' and hidden_matrix(name, parameter) else 2
        optimizer_bytes += moments * parameter.numel() * parameter.element_size()
        if moments == 2:
            step_scalars += max(8, torch.empty((), device='meta').element_size()) * world
        if distributed and parameter.ndim:
            # Upper allowance for uneven first-dimension shard padding, including
            # model and moments. Scalars conservatively replicated below.
            row = math.prod(parameter.shape[1:]) * parameter.element_size()
            shard_padding += (world - 1) * row * (1 + moments)
        elif distributed:
            shard_padding += (world - 1) * parameter.numel() * parameter.element_size() * (1 + moments)
    portable_parameter_bytes = parameter_bytes
    if distributed:
        # DCP addresses tensors by FQN; do not assume cross-key tied-storage
        # deduplication as torch.save provides for ordinary portable checkpoints.
        all_parameters = list(model.named_parameters(remove_duplicate=False))
        parameter_bytes = sum(p.numel() * p.element_size() for _, p in all_parameters)
        unique_ids = set()
        for _, parameter in all_parameters:
            if id(parameter) in unique_ids:
                row = math.prod(parameter.shape[1:]) * parameter.element_size() if parameter.ndim else parameter.element_size()
                shard_padding += (world - 1) * row
            unique_ids.add(id(parameter))
    # state_dict selects persistent buffers; backing storage aliases deduplicated.
    buffer_storages = {}
    parameter_storages = {p.untyped_storage()._cdata for p in model.parameters()}
    for tensor in model.state_dict().values():
        storage = tensor.untyped_storage()
        if storage._cdata not in parameter_storages:
            buffer_storages[storage._cdata] = storage.nbytes()
    buffers = sum(buffer_storages.values()) * (world if distributed else 1)
    tensor_bytes = parameter_bytes + optimizer_bytes + step_scalars + shard_padding + buffers
    def estimate(tensors, copies):
        base = tensors + _METADATA_ALLOWANCE * copies
        return base + (base + 9) // 10 + _SERIALIZATION_OVERHEAD * copies
    full = estimate(tensor_bytes, world if distributed else 1)
    portable = estimate(portable_parameter_bytes + sum(buffer_storages.values()), 1) if export_weights else 0
    counts = {'initial': int(retention['save_initial_checkpoint']), 'final': 1, 'milestones': milestone_count,
              'separate_90_percent_branch': separate_branch,
              'rolling_recovery_including_staging': rolling}
    count = sum(counts.values())
    return {'kind': 'checkpoint_storage_projection', 'required_additional_bytes': full * count + portable,
            'per_checkpoint_estimate_bytes': full, 'export_weights_estimate_bytes': portable,
            'checkpoint_count': count, 'count_breakdown': counts, 'milestone_policy': policy,
            'checkpoint_retention_policy': retention,
            'start_tokens': start, 'token_limit': limit, 'reachable_tokens': reachable,
            'reachable_updates': updates, 'distributed': distributed, 'world_size': world,
            'dtype': str(torch.get_default_dtype()),
            'tensor_bytes_breakdown': {'parameters': parameter_bytes, 'optimizer_moments': optimizer_bytes,
                'optimizer_step_scalars': step_scalars, 'persistent_buffers': buffers,
                'distributed_shard_padding_allowance': shard_padding},
            'metadata_allowance_bytes_per_rank_checkpoint': _METADATA_ALLOWANCE,
            'scope': 'advisory projection, not guaranteed upper bound; no reservation or deletion',
            'assumptions': ['mature optimizer state even for initial checkpoint',
                'scheduled/time-based recovery may occur every reachable update',
                'one extra recovery staging copy before pruning when recovery saves are enabled',
                'persistent distributed buffers conservatively replicated on every rank',
                'DCP model parameter aliases counted separately by FQN; portable exports deduplicate storage',
                '10% serialization margin plus 1MiB overhead per rank checkpoint',
                'existing artifacts and unbounded concurrent pinned old recoveries excluded',
                'arbitrary metadata/code beyond fixed allowance and filesystem allocation excluded',
                'optional export is additional weight-only artifact; time caps may reduce updates']}
