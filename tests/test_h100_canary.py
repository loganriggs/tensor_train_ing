"""Canary metrics use global tokens and summed slowest-rank update durations."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('h100_canary', Path(__file__).resolve().parents[1] / 'scripts/h100_canary.py')
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


def fixture():
    rows = [dict(step=i, loss=3., seconds=2., prediction_tokens=100, score_bound_max=float(i)) for i in range(1, 21)]
    result = dict(progress=dict(steps=20, tokens=2000), parameters=dict(total=1_000_000_000), checkpoint='fresh/train/final')
    kwargs = dict(steps=20, world=2, peak_tflops_per_gpu=100., memory=[dict(rank=0), dict(rank=1)], wall_seconds=50.)
    return result, rows, kwargs


def test_canary_rates_and_mfu():
    result, rows, kwargs = fixture()
    report = canary.summarize(result, rows, **kwargs)
    assert report['all_updates']['tokens_per_second'] == 50
    assert report['all_updates']['tokens_per_second_per_gpu'] == 25
    assert report['all_updates']['mfu_estimate'] == pytest.approx(.0015)
    assert report['after_startup_updates']['steps'] == 10
    assert report['end_to_end_tokens_per_second'] == 40
    assert report['score_bound_max'] == 20


@pytest.mark.parametrize('field,value', [('loss', float('nan')), ('score_bound_max', float('inf')), ('seconds', 0), ('prediction_tokens', -1), ('step', 99)])
def test_bad_metrics_fail(field, value):
    result, rows, kwargs = fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        canary.summarize(result, rows, **kwargs)


def test_partial_run_is_failure():
    result, rows, kwargs = fixture()
    with pytest.raises(ValueError, match='incomplete'):
        canary.summarize(result, rows[:-1], **kwargs)


def test_missing_bound_cannot_pass():
    result, rows, kwargs = fixture()
    del rows[0]['score_bound_max']
    with pytest.raises(KeyError):
        canary.summarize(result, rows, **kwargs)
