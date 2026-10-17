"""Ordered, report-cached validation comparisons without model execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .evaluate import compare_evaluations


def compare_evaluation_pairs(pairs, out_dir, *, bootstrap_samples=1000, seed=0):
    """Validate every pair before publishing one immutable comparison bundle.

    Names are arbitrary nonempty labels, never filesystem path components. Each
    resolved report is read exactly once and bound by the SHA256 of those bytes.
    Every result retains the existing comparator's paired document uncertainty;
    the bundle adds no pooled estimate, ranking or multiplicity correction.
    """
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('pairs must be a nonempty list')
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 100_000:
        raise ValueError('bootstrap_samples must be in [0,100000]')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    names, cache, inputs, results = set(), {}, {}, []
    def read(value):
        if not isinstance(value, (str, Path)):
            raise ValueError('comparison inputs must be report paths')
        path = str(Path(value).resolve())
        if path not in cache:
            raw = Path(path).read_bytes()
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError('evaluation report must be a JSON object')
            cache[path] = report
            inputs[path] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return path, cache[path]
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {'name', 'left', 'right'}:
            raise ValueError('each pair requires exactly name, left and right')
        name = pair['name']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('comparison names must be unique nonempty labels')
        names.add(name)
        left_path, left = read(pair['left'])
        right_path, right = read(pair['right'])
        result = compare_evaluations(left, right, None, bootstrap_samples=bootstrap_samples, seed=seed)
        results.append({'name': name, 'left': left_path, 'right': right_path, 'result': result})
    report = {'kind': 'batched_validation_comparisons', 'pairs': results, 'inputs': inputs,
              'bootstrap_samples': bootstrap_samples, 'seed': seed,
              'automatic_ranking_eligible': False,
              'scope': 'ordered, separately computed matched-target comparisons; nominal paired document intervals, not training-seed variance; no pooled estimate, automatic ranking or multiplicity correction'}
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    pending = out / '.comparisons.json.tmp'
    with pending.open('w') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, out / 'comparisons.json')
    descriptor = os.open(out, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return report
