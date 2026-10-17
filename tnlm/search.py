"""Deterministic explicit sweep planning, without launching or reserving runs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping

from .config import Config, ModelConfig, TrainConfig
from .runtime import full_config
from .train import plan_search

_ALLOWED = {'model', 'train', 'project', 'data', 'evaluation'}
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if not isinstance(key, str):
            raise ValueError('override keys must be strings')
        if isinstance(value, Mapping):
            if key in result and not isinstance(result[key], Mapping):
                raise ValueError(f'cannot merge mapping into scalar field {key}')
            result[key] = _merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def plan_sweep(base_cfg, candidates) -> dict:
    """Resolve an ordered explicit candidate list for fresh uncooled trajectories.

    Each candidate is exactly {'name': safe path component, 'overrides': mapping}.
    Costs/storage are existing plan_search projections for full configured rungs;
    separate cooldown, evaluation, and concurrent pinned artifacts are excluded.
    No model execution, ledger writes, reservations or directory creation occur.
    """
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('candidates must be a nonempty list')
    base = full_config(base_cfg)
    names = set()
    runs = []
    known_cost = 0.
    known_seconds = 0.
    missing_cost = []
    missing_seconds = []
    storage = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or set(candidate) != {'name', 'overrides'}:
            raise ValueError('each candidate requires exactly name and overrides')
        name = candidate['name']
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in ('.', '..'):
            raise ValueError('candidate name must be a safe path component')
        if name in names:
            raise ValueError('candidate names must be unique')
        names.add(name)
        override = candidate['overrides']
        if not isinstance(override, Mapping) or set(override) - _ALLOWED:
            raise ValueError('unknown top-level override keys or nonmapping overrides')
        if any(not isinstance(value, Mapping) for value in override.values()):
            raise ValueError('top-level configuration overrides must be mappings')
        resolved = _merge(base, override)
        try:
            cfg = Config(model=ModelConfig(**resolved['model']), train=TrainConfig(**resolved['train']),
                         raw=resolved)
            config = full_config(cfg)
            encoded = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
            plan = plan_search(cfg)
        except (TypeError, AssertionError, OverflowError) as exc:
            raise ValueError(f'invalid candidate configuration: {name}') from exc
        projected = plan['proposed_runs']
        cost_values = [run['projected_usd'] for run in projected]
        seconds_values = [run['projected_seconds'] for run in projected]
        if any(value is None for value in cost_values):
            missing_cost.append(name)
        else:
            known_cost += sum(cost_values)
        if any(value is None for value in seconds_values):
            missing_seconds.append(name)
        else:
            known_seconds += sum(seconds_values)
        storage += plan['checkpoint_storage_plan']['required_additional_bytes']
        runs.append({'name': name, 'config': config,
                     'config_sha256': hashlib.sha256(encoded.encode()).hexdigest(), 'plan': plan})
    if not math.isfinite(known_cost) or not math.isfinite(known_seconds):
        raise ValueError('aggregate projected cost/time must be finite')
    return {'kind': 'frozen_explicit_sweep_plan', 'runs': runs, 'candidate_count': len(runs),
            'required_additional_bytes': storage,
            'projected_usd': None if missing_cost else known_cost,
            'projected_seconds': None if missing_seconds else known_seconds,
            'cost_projection_complete': not missing_cost,
            'known_projected_usd': known_cost, 'unpriced_candidates': missing_cost,
            'untimed_candidates': missing_seconds,
            'scope': 'fresh uncooled full-rung trajectories only; excludes separate cooldown/evaluation; no execution or reservation',
            'storage_scope': 'sum of authoritative per-run advisory checkpoint projections; concurrent pins/existing artifacts excluded'}
