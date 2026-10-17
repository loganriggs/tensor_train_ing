"""Strict metadata-based paired-seed validation summaries; no model execution."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from .evaluate import compare_evaluations, _training_comparison


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a SHA256 hex digest')
    return value


_CONFIG_DIMENSIONS = {
    'optimizer': 'train.optimizer', 'width': 'model.d_model', 'depth': 'model.layers',
    'train_context': 'data.context', 'rung_total_tokens': 'train.rung_total_tokens',
    'global_prediction_tokens_per_update': 'train.global_prediction_tokens_per_update',
    'lr': 'train.lr', 'normalization': 'model.normalization', 'attention': 'model.attention',
    'aux_adamw_lr': 'train.aux_adamw_lr', 'schedule': 'train.schedule',
    **{key: 'model.' + key for key in ('width_scaling_base', 'width_lr_exponent', 'width_logit_exponent',
        'depth_scaling_base', 'depth_lr_exponent', 'depth_residual_base', 'depth_residual_exponent', 'residual_alpha')},
    **{key: 'train.' + key for key in ('weight_decay_hidden', 'beta1', 'beta2')},
}


def _configuration(report, seed):
    value = report.get('training_descriptor', {}).get('configuration')
    if (not isinstance(value, dict) or not isinstance(value.get('model'), dict)
            or not isinstance(value.get('train'), dict) or 'project' in value
            or type(value['train'].get('seed')) is not int or value['train']['seed'] != seed):
        raise ValueError('complete checkpoint configuration with matching seed is required')
    return value


def _leaves(value, prefix='config'):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or '.' in key:
                raise ValueError('configuration keys must be unambiguous dot-path components')
            result.update(_leaves(item, prefix + '.' + key))
        # Keep empty mappings as explicit known values.
        return result if value else {prefix: {}}
    return {prefix: value}


def _configuration_differences(left, right, *, ignore_seed=False):
    left, right = _leaves(left), _leaves(right)
    if set(left) != set(right):
        raise ValueError('unknown/missing full configuration evidence paths')
    return [key for key in sorted(left) if not (ignore_seed and key == 'config.train.seed')
            and (type(left[key]) is not type(right[key]) or left[key] != right[key])]


def compare_seed_pairs(pairs, out_dir, *, allowed_between_dimensions) -> dict:
    """Compare complete, matched-target conditions using equal weights per seed.

    Input reports are read once and bound by their exact byte SHA256. Their
    checkpoint metadata is evidence, not verification of prospective declaration
    or independent training. A shared numeric seed does not prove coupled RNG.
    Document bootstrap intervals are not seed uncertainty; no seed CI/p-value or
    winner is inferred. All validation precedes output-directory creation.
    """
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError('at least two explicit seed pairs are required')
    if (not isinstance(allowed_between_dimensions, list)
            or any(not isinstance(value, str) for value in allowed_between_dimensions)
            or len(set(allowed_between_dimensions)) != len(allowed_between_dimensions)):
        raise ValueError('allowed_between_dimensions must be a unique list of dimension names')
    if 'seed' in allowed_between_dimensions:
        raise ValueError('seed cannot be an allowed between-condition difference')
    cache, seeds = {}, set()
    duplicate_model_outcomes = {'left': [], 'right': []}
    references = {}
    identities = {'left': {'model': set(), 'state': set(), 'manifest': set()},
                  'right': {'model': set(), 'state': set(), 'manifest': set()}}
    cross_condition_duplicates = []
    baseline = None
    per_seed = []
    def read(value):
        if not isinstance(value, (str, Path)):
            raise ValueError('seed-pair inputs must be evaluation report paths')
        path = Path(value).resolve()
        if path not in cache:
            raw = path.read_bytes()
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError('evaluation report must be a JSON object')
            cache[path] = report, hashlib.sha256(raw).hexdigest()
        report, digest = cache[path]
        return report, str(path), digest
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {'seed', 'left', 'right'}:
            raise ValueError('each pair requires exactly seed, left and right')
        seed = pair['seed']
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError('pair seeds must be unique nonnegative integers')
        seeds.add(seed)
        loaded = {side: read(pair[side]) for side in ('left', 'right')}
        reports = {side: value[0] for side, value in loaded.items()}
        comparison = compare_evaluations(reports['left'], reports['right'], None, bootstrap_samples=0)
        dimensions = comparison['training_comparison']['dimensions']
        configurations = {side: _configuration(reports[side], seed) for side in ('left', 'right')}
        paths = set(_leaves(configurations['left'])) & set(_leaves(configurations['right']))
        if any(value not in dimensions and value not in paths for value in allowed_between_dimensions):
            raise ValueError('unknown allowed training dimension or configuration path')
        if 'config.train.seed' in allowed_between_dimensions:
            raise ValueError('seed cannot be an allowed between-condition difference')
        full_differences = _configuration_differences(configurations['left'], configurations['right'])
        mapped = {'config.' + _CONFIG_DIMENSIONS[name] for name in allowed_between_dimensions if name in _CONFIG_DIMENSIONS}
        if set(full_differences) - set(allowed_between_dimensions) - mapped:
            raise ValueError('undeclared full configuration differences: ' + ', '.join(sorted(set(full_differences) - set(allowed_between_dimensions) - mapped)))
        unknown = [name for name, value in dimensions.items() if value['status'] == 'unknown']
        if unknown:
            raise ValueError('unknown training evidence dimensions: ' + ', '.join(unknown))
        forbidden = [name for name, value in dimensions.items()
                     if value['status'] == 'different' and name not in allowed_between_dimensions]
        if forbidden:
            raise ValueError('undeclared between-condition differences: ' + ', '.join(forbidden))
        entry = {'seed': seed, 'comparison': comparison, 'full_configuration_differences': full_differences}
        for side in ('left', 'right'):
            report, path, report_sha = loaded[side]
            descriptor = report.get('training_descriptor', {})
            if type(descriptor.get('seed')) is not int or descriptor['seed'] != seed:
                raise ValueError('checkpoint descriptor seed differs from declared pair seed')
            if descriptor.get('completion_status') != 'complete':
                raise ValueError('seed comparison requires complete training segments')
            for key in ('training_manifest_sha256', 'validation_manifest_sha256', 'target_plan_sha256', 'available_prefix_plan_sha256'):
                _hash(report.get(key), key)
            if type(report.get('evaluation_seed')) is not int or report['evaluation_seed'] < 0:
                raise ValueError('known nonnegative evaluation seed is required')
            signature = {key: report.get(key) for key in ('target_plan', 'target_plan_sha256',
                         'validation_manifest_sha256', 'training_manifest_sha256', 'evaluation_seed', 'context', 'available_prefix_plan_sha256')}
            if baseline is None:
                baseline = signature
            elif signature != baseline:
                raise ValueError('all seed reports require identical validation targets, context and prefixes')
            if side in references:
                full_within = _configuration_differences(_configuration(references[side], references[side]['training_descriptor']['seed']), configurations[side], ignore_seed=True)
                if full_within:
                    raise ValueError('within-condition full configuration differs: ' + ', '.join(full_within))
                same = _training_comparison(references[side], report)['dimensions']
                mismatches = [name for name, value in same.items()
                              if name != 'seed' and value['status'] != 'matched']
                if mismatches:
                    raise ValueError('within-condition training identity differs or is unknown: ' + ', '.join(mismatches))
            else:
                references[side] = report
            model_sha = _hash(report.get('model_sha256'), 'model_sha256')
            state_sha = _hash(report.get('checkpoint_state_sha256'), 'checkpoint_state_sha256')
            manifest_sha = _hash(report.get('checkpoint_manifest_sha256'), 'checkpoint_manifest_sha256')
            for key, value in [('model', model_sha), ('state', state_sha), ('manifest', manifest_sha)]:
                if value in identities[side][key]:
                    if key != 'model':
                        raise ValueError('duplicate checkpoint identity within condition')
                    duplicate_model_outcomes[side].append({'seed': seed, 'model_sha256': value})
                identities[side][key].add(value)
            entry[side] = {'report_path': path, 'report_sha256': report_sha,
                          'model_sha256': model_sha, 'checkpoint_state_sha256': state_sha,
                          'checkpoint_manifest_sha256': manifest_sha,
                          'checkpoint': report.get('checkpoint'), 'training_descriptor': descriptor}
        per_seed.append(entry)
    for key in ('model', 'state', 'manifest'):
        overlap = identities['left'][key] & identities['right'][key]
        if overlap:
            cross_condition_duplicates.append({'identity': key, 'sha256': sorted(overlap)})
    deltas = [row['comparison']['delta_ce_right_minus_left'] for row in per_seed]
    left_ce = [row['comparison']['left']['ce'] for row in per_seed]
    right_ce = [row['comparison']['right']['ce'] for row in per_seed]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in deltas + left_ce + right_ce):
        raise ValueError('per-seed CE and deltas must be finite')
    count = len(per_seed)
    deviation = statistics.stdev(deltas)
    error = abs(deltas[0] - deltas[1]) / 2 if count == 2 else deviation / math.sqrt(count)
    tokens = per_seed[0]['comparison']['prediction_tokens']
    report = {'kind': 'paired_seed_validation_comparison',
        'allowed_between_dimensions': list(allowed_between_dimensions), 'per_seed': per_seed,
        'aggregate': {'seeds': count, 'mean_ce_left': statistics.mean(left_ce),
                      'mean_ce_right': statistics.mean(right_ce),
                      'mean_paired_delta_ce': statistics.mean(deltas),
                      'paired_delta_sample_sd': deviation, 'paired_delta_standard_error': error,
                      'weighting': 'equal weight per seed; each CE is token-weighted on identical validation targets'},
        'training_identity': {'left': references['left']['training_descriptor'],
                              'right': references['right']['training_descriptor'],
                              'reference_seed_scope': 'first declared seed is representative only of matched non-seed dimensions'},
        'validation': {'manifest_sha256': baseline['validation_manifest_sha256'],
                       'target_plan_sha256': baseline['target_plan_sha256'],
                       'context': baseline['context'],
                       'available_prefix_plan_sha256': baseline['available_prefix_plan_sha256'],
                       'unique_prediction_tokens': tokens,
                       'repeated_prediction_tokens_per_condition': count * tokens,
                       'total_scored_prediction_tokens_across_conditions': 2 * count * tokens,
                       'scope': 'repeated scoring of one frozen validation target set; repetitions do not create new validation targets'},
        'cross_condition_duplicate_identities': cross_condition_duplicates,
        'duplicate_model_outcomes': duplicate_model_outcomes,
        'unique_model_count': {side: len(identities[side]['model']) for side in ('left', 'right')},
        'configuration_identity_scope': 'all checkpoint configuration sections except project; within-condition only train.seed differs; all between-condition summary/full configuration changes explicitly declared',
        'automatic_ranking_eligible': False,
        'uncertainty_scope': 'sample SD/SE across observed seed-pair deltas; no seed confidence interval or p-value; document intervals are not seed variance',
        'limitations': ['same numeric seed does not establish coupled random streams',
                       'reports do not establish prospective declaration, run independence or absence of HPO selection',
                       'checkpoint hashes are report-bound identities, not revalidated checkpoint contents',
                       'configuration equality does not certify runtime/code equality; per-seed comparison probe provenance retains available source evidence',
                       'complete means the declared training segment endpoint; no automatic winner or promotion']}
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    pending = out / '.replication.json.tmp'
    with pending.open('w') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, out / 'replication.json')
    descriptor = os.open(out, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return report
