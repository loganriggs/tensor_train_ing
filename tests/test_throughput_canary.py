"""CPU-only gates over completed real updates, with fail-closed accounting."""
import json

import numpy as np
import pytest
import torch

from helpers import state
from tnlm.throughput_canary import ThroughputCanary
from tnlm.train import train_run


@pytest.mark.parametrize('steps,threshold,warmup', [
    (True, 1., 0), (0, 1., 0), (1.5, 1., 0),
    (1, True, 0), (1, 0., 0), (1, float('nan'), 0), (1, float('inf'), 0),
    (1, 1., True), (1, 1., -1),
])
def test_invalid_controls_create_nothing(tmp_path, steps, threshold, warmup):
    with pytest.raises(ValueError):
        ThroughputCanary(steps, threshold, warmup, output_path=tmp_path / 'canary.json')
    assert not list(tmp_path.iterdir())


def test_ratio_of_sums_warmup_and_bounded_selection(tmp_path):
    path = tmp_path / 'canary.json'
    canary = ThroughputCanary(2, 24., 1, output_path=path)
    assert canary.required_updates == 3
    assert not path.exists()
    canary.observe({'step': 1})  # Warmup has no measured timing requirement.
    assert not path.exists()
    canary.observe({'step': 2, 'prediction_tokens': 10, 'seconds': 1.})
    assert json.loads(path.read_text())['status'] == 'collecting'
    canary.observe({'step': 3, 'prediction_tokens': 90, 'seconds': 3.})
    report = canary.require_complete()
    assert report['tokens_per_second'] == 25.  # Mean of per-step rates is20, wrong.
    assert report['prediction_tokens'] == 100
    assert report['status'] == 'passed'
    canary.observe({'step': 4, 'prediction_tokens': 1, 'seconds': 1000.})
    assert canary.report() == report
    assert len(report['selected_steps']) == 2
    assert json.loads(path.read_text()) == report
    assert not list(tmp_path.glob('.canary.json.*'))


def test_low_and_insufficient_gate_evidence(tmp_path):
    low = ThroughputCanary(1, 11., 0, output_path=tmp_path / 'low.json')
    with pytest.raises(RuntimeError, match='below'):
        low.observe({'step': 1, 'prediction_tokens': 10, 'seconds': 1.})
    assert json.loads((tmp_path / 'low.json').read_text())['status'] == 'below_threshold'
    short = ThroughputCanary(2, 1., 1, output_path=tmp_path / 'short.json')
    short.observe({'step': 1})
    with pytest.raises(RuntimeError, match='insufficient'):
        short.require_complete()
    report = json.loads((tmp_path / 'short.json').read_text())
    assert report['status'] == 'insufficient'
    assert report['tokens_per_second'] is None


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def test_real_passing_canary_does_not_retrain_or_change_state(config, tmp_path):
    baseline = train_run(config, tmp_path / 'baseline', max_steps=3, max_seconds=60, device='cpu')
    canary = ThroughputCanary(2, 1e-12, 1)
    measured = train_run(config, tmp_path / 'measured', max_steps=3, max_seconds=60,
                         device='cpu', _throughput_canary=canary)
    a, b = state(tmp_path / 'baseline/final'), state(tmp_path / 'measured/final')
    for key in ('model', 'optimizer', 'rng', 'sampler', 'progress'):
        equal(a[key], b[key])
    assert baseline['progress']['steps'] == measured['progress']['steps'] == 3
    report = measured['throughput_canary']
    assert report['status'] == 'passed'
    assert [row['step'] for row in report['selected_steps']] == [2, 3]
    assert report['prediction_tokens'] == 14
    assert report['training_seconds'] == sum(row['seconds'] for row in measured['history'][1:])
    assert json.loads((tmp_path / 'measured/canary.json').read_text()) == report


@pytest.mark.parametrize('kind', ['low', 'insufficient'])
def test_real_failed_gate_finalizes_bill_and_preserves_metrics(config, tmp_path, kind):
    canary = ThroughputCanary(1 if kind == 'low' else 2, 1e100 if kind == 'low' else 1e-12, 0)
    directory = tmp_path / kind
    with pytest.raises(RuntimeError, match='canary'):
        train_run(config, directory, max_steps=1, max_seconds=60,
                  device='cpu', _throughput_canary=canary)
    bill = json.loads((tmp_path / 'spend.json').read_text())['runs'][str(directory.resolve())]
    assert bill['status'] == 'failed' and bill['reserved_usd'] == 0
    assert bill['elapsed_seconds'] > 0
    assert (directory / 'initial/manifest.json').exists()
    assert not (directory / 'final').exists()
    assert len((directory / 'metrics.jsonl').read_text().splitlines()) == 1
    assert json.loads((directory / 'canary.json').read_text())['status'] == (
        'below_threshold' if kind == 'low' else 'insufficient')
