"""Training result evidence must equal the authoritative final checkpoint identity."""
import json
from pathlib import Path

import torch

from tnlm.evaluate import _training_descriptor
from tnlm.train import train_run


def assert_identity(result, *, cooldown, completion, tokens):
    payload = torch.load(Path(result['checkpoint']) / 'state.pt', weights_only=False,
                         map_location='cpu', mmap=True)
    expected = _training_descriptor(payload)
    assert result['training_descriptor'] == expected
    persisted = json.loads((Path(result['checkpoint']).parent / 'result.json').read_text())
    assert persisted['training_descriptor'] == expected
    assert expected['cooldown_status'] == cooldown
    assert expected['completion_status'] == completion
    assert expected['actual_prediction_tokens'] == tokens
    return expected


def test_real_cpu_full_partial_and_exact_cooldown_result_identity(config, tmp_path):
    config.train.rung_total_tokens = 31
    config.raw['project']['milestone_tokens'] = []
    partial = train_run(config, tmp_path / 'partial', max_steps=2, max_seconds=60, device='cpu')
    partial_descriptor = assert_identity(partial, cooldown='uncooled', completion='partial', tokens=14)
    full = train_run(config, tmp_path / 'full', max_steps=5, max_seconds=60, device='cpu')
    full_descriptor = assert_identity(full, cooldown='uncooled', completion='complete', tokens=31)
    branch = tmp_path / 'full' / 'milestone-000000000027'
    assert branch.is_dir()
    cooled = train_run(config, tmp_path / 'cooled', max_steps=1, max_seconds=60, device='cpu',
                       resume=branch, cooldown_steps=1, cooldown_end_tokens=31)
    descriptor = assert_identity(cooled, cooldown='cooled', completion='complete', tokens=31)
    assert descriptor['cooldown_start_tokens'] == 28
    assert descriptor['cooldown_steps'] == descriptor['scheduler_step'] == 1
    assert descriptor['segment_target_prediction_tokens'] == 31
    assert descriptor['warmup_steps'] == full_descriptor['warmup_steps'] == partial_descriptor['warmup_steps']
    assert descriptor['train_context'] == config.raw['data']['context']
    assert descriptor['optimizer'] == config.train.optimizer
