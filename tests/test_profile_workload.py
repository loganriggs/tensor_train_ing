"""Profile accounting distinguishes prediction targets from executed work."""
import importlib
import json

import pytest


@pytest.mark.parametrize('warmup', [0, 1, 3])
def test_profile_work_counts_follow_same_timed_window(config, tmp_path, monkeypatch, warmup):
    training = importlib.import_module('tnlm.train')
    run = tmp_path / 'profile'
    run.mkdir()
    rows = [
        {'seconds': 10., 'prediction_tokens': 8, 'microbatches': 2,
         'forward_input_positions': 16, 'unscored_rows_omitted': 0},
        {'seconds': 2., 'prediction_tokens': 8, 'microbatches': 2,
         'forward_input_positions': 10, 'unscored_rows_omitted': 3},
        {'seconds': 1., 'prediction_tokens': 3, 'microbatches': 1,
         'forward_input_positions': 4, 'unscored_rows_omitted': 1},
    ]
    metrics = run / 'metrics.jsonl'
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'runs': {str(run.resolve()): {'status': 'complete'}}}))
    config.raw['project']['spend_ledger'] = str(ledger)
    payload = dict(metrics_path=str(metrics), cuda_peak_allocated_bytes=None,
                   cuda_peak_reserved_bytes=None, checkpoint='fixture', operator_audit={},
                   parameters=100, compile_enabled=True, stop_reason='token_limit')
    monkeypatch.setattr(training, 'train_run', lambda *args, **kwargs: payload)
    result = training.profile_run(config, run, steps=3, warmup_steps=warmup,
                                  max_seconds=60, device='cpu')
    selected = rows[warmup:]
    tokens = sum(row['prediction_tokens'] for row in selected)
    positions = sum(row['forward_input_positions'] for row in selected)
    micros = sum(row['microbatches'] for row in selected)
    assert result['timed_prediction_tokens'] == tokens
    assert result['timed_forward_input_positions'] == positions
    assert result['timed_microbatches'] == micros
    assert result['timed_unscored_rows_omitted'] == sum(row['unscored_rows_omitted'] for row in selected)
    assert result['forward_positions_per_prediction_target'] == (positions / tokens if tokens else None)
    assert result['mean_microbatches_per_step'] == (micros / len(selected) if selected else None)
    assert 'recompilation' in result['note']
    assert json.loads((run / 'profile.json').read_text()) == result
