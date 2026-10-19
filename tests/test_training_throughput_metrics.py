"""Wall-clock row rates use segment-local tokens and include inter-row overhead."""
import json
import time
from pathlib import Path

import pytest
import torch.distributed as dist

from tnlm.distributed_train import distributed_train_run
from tnlm.train import train_run


def _rows(path):
    return [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]


def _rates(rows, start_tokens=0):
    previous_tokens, previous_elapsed = start_tokens, 0.
    for row in rows:
        elapsed = row['run_elapsed_seconds'] - previous_elapsed
        tokens = row['tokens'] - previous_tokens
        assert tokens == row['prediction_tokens']
        assert row['tokens_per_second'] == pytest.approx(tokens / elapsed, rel=1e-15)
        assert row['tokens_per_second'] > 0 and row['seconds'] > 0
        previous_tokens, previous_elapsed = row['tokens'], row['run_elapsed_seconds']


@pytest.mark.parametrize('distributed', [False, True])
def test_row_throughput_and_resume_use_new_segment_tokens(config, tmp_path, distributed):
    config.train.rung_total_tokens = 17  # Final update is only three targets.
    config.raw['project'].update(history_limit=1, keep_recovery_checkpoints=0, milestone_tokens=[])
    trainer = distributed_train_run if distributed else train_run
    if distributed:
        dist.init_process_group('gloo', init_method=f'file://{tmp_path / "group"}', rank=0, world_size=1)
    try:
        whole = trainer(config, tmp_path / 'whole', max_steps=3, max_seconds=60, device='cpu')
        first = trainer(config, tmp_path / 'first', max_steps=1, max_seconds=60, device='cpu')
        resumed = trainer(config, tmp_path / 'resumed', max_steps=2, max_seconds=60,
                          device='cpu', resume=first['checkpoint'])
    finally:
        if distributed:
            dist.destroy_process_group()
    rows, resumed_rows = _rows(tmp_path / 'whole'), _rows(tmp_path / 'resumed')
    assert [row['prediction_tokens'] for row in rows] == [7, 7, 3]
    assert len(whole['history']) == len(resumed['history']) == 1
    _rates(rows)
    _rates(resumed_rows, start_tokens=7)
    assert resumed_rows[0]['tokens'] == 14
    assert resumed_rows[0]['tokens_per_second'] == pytest.approx(7 / resumed_rows[0]['run_elapsed_seconds'])


def test_row_throughput_includes_checkpoint_overhead_without_changing_step_timer(config, tmp_path, monkeypatch):
    import tnlm.checkpoint as checkpoints
    config.raw['project'].update(checkpoint_every_steps=1, keep_recovery_checkpoints=1,
                                 milestone_tokens=[], history_limit=1)
    original = checkpoints.save_checkpoint

    def slow_recovery(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if Path(path).name == 'recovery-00000001':
            time.sleep(.05)
        return result

    monkeypatch.setattr(checkpoints, 'save_checkpoint', slow_recovery)
    train_run(config, tmp_path / 'slow-save', max_steps=2, max_seconds=60, device='cpu')
    rows = _rows(tmp_path / 'slow-save')
    _rates(rows)
    interval = rows[1]['run_elapsed_seconds'] - rows[0]['run_elapsed_seconds']
    assert interval >= rows[1]['seconds'] + .05
    assert rows[1]['tokens_per_second'] < rows[1]['prediction_tokens'] / rows[1]['seconds']
