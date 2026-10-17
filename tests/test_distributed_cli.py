"""Distributed command admission and ownership of the process-group lifecycle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch.distributed as dist
import yaml

from tnlm.cli import _distributed_run_from_cli


def args(tmp_path):
    return argparse.Namespace(collective_timeout_seconds=30., device='cpu',
        run_dir=str(tmp_path / 'run'), max_steps=2, max_seconds=60., resume=None,
        run_budget_usd=None)


def test_distributed_dry_run_does_not_initialize_group(config, tmp_path, monkeypatch):
    config.raw['project']['dry_run'] = True
    monkeypatch.setattr(dist, 'init_process_group', lambda *a, **kw: pytest.fail('dry run initialized group'))
    monkeypatch.setenv('RANK', '0')
    result = _distributed_run_from_cli(config, args(tmp_path))
    assert result['dry_run'] and result['distributed']
    monkeypatch.setenv('RANK', '1')
    assert _distributed_run_from_cli(config, args(tmp_path)) is None
    assert not (tmp_path / 'spend.json').exists()


def test_distributed_requires_launch_before_admission(config, tmp_path, monkeypatch):
    monkeypatch.setattr(dist, 'is_initialized', lambda: False)
    monkeypatch.delenv('MASTER_PORT', raising=False)
    with pytest.raises(ValueError, match='torchrun launch'):
        _distributed_run_from_cli(config, args(tmp_path))
    assert not (tmp_path / 'spend.json').exists()


@pytest.mark.parametrize('existing,rank,fail', [(False, 0, False), (False, 1, False),
                                            (False, 0, True), (True, 0, True)])
def test_group_cleanup_and_single_coordinator_result(config, tmp_path, monkeypatch, existing, rank, fail):
    events = []
    for name, value in dict(RANK=rank, WORLD_SIZE=2, LOCAL_RANK=rank,
                            MASTER_ADDR='localhost', MASTER_PORT=12345).items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(dist, 'is_initialized', lambda: existing)
    monkeypatch.setattr(dist, 'get_rank', lambda: rank)
    def initialize(backend, *, timeout):
        assert backend == 'gloo' and timeout.total_seconds() == 30
        events.append('init')
    monkeypatch.setattr(dist, 'init_process_group', initialize)
    monkeypatch.setattr(dist, 'destroy_process_group', lambda: events.append('destroy'))
    def run(cfg, path, **kwargs):
        assert kwargs['device'] == 'cpu' and kwargs['max_steps'] == 2
        events.append('train')
        if fail:
            raise ValueError('training failure')
        return {'done': True}
    fake = types.ModuleType('tnlm.distributed_train')
    fake.distributed_train_run = run
    monkeypatch.setitem(sys.modules, 'tnlm.distributed_train', fake)
    if fail:
        with pytest.raises(ValueError, match='training failure'):
            _distributed_run_from_cli(config, args(tmp_path))
    else:
        assert _distributed_run_from_cli(config, args(tmp_path)) == ({'done': True} if rank == 0 else None)
    assert events == (['train'] if existing else ['init', 'train', 'destroy'])


@pytest.mark.slow
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_actual_torchrun_cli_creates_one_billed_trajectory(config, tmp_path, optimizer):
    config.train.optimizer = optimizer
    config.train.aux_adamw_lr = .002
    from tnlm.data import prepare_data
    from helpers import FixtureTokenizer
    source = tmp_path / 'heldout.jsonl'
    source.write_text(json.dumps({'id': 'heldout', 'text': 'opqrstuvwxyz'}) + '\n')
    manifest = prepare_data(tmp_path / 'heldout', source_jsonl=source, revision='fixture-v1',
        tokenizer=FixtureTokenizer(), split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    config.raw['evaluation'] = {'validation_manifest': str(manifest)}
    path = tmp_path / 'run.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    run_dir = tmp_path / 'distributed-cli'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', '2',
               '-m', 'tnlm.cli', 'train', '--distributed', '--config', str(path),
               '--run-dir', str(run_dir), '--max-steps', '1', '--max-seconds', '60',
               '--collective-timeout-seconds', '30', '--device', 'cpu',
               '--export-max-state-bytes', '10000000']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-6000:]
    result = json.loads((run_dir / 'result.json').read_text())
    assert result['world_size'] == 2 and result['progress']['steps'] == 1
    assert result['progress']['tokens'] == 7
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    assert list(ledger['runs']) == [str(run_dir.resolve())]
    assert result['billing']['status'] == 'complete'
    assert result['billing']['reserved_usd'] == 0
    assert (run_dir / 'initial' / 'manifest.json').exists()
    assert (run_dir / 'final' / 'manifest.json').exists()

    from tnlm.checkpoint import load_checkpoint
    from tnlm.model import TensorLM
    portable = load_checkpoint(result['evaluation_checkpoint'], TensorLM(config.model), restore_rng=False)
    assert portable['progress']['steps'] == 1
    assert portable['optimizer'] is None and portable['extra']['training_recovery'] is False
    import hashlib
    assert portable['parent'] == {'path': str(run_dir / 'final'),
        'manifest_sha256': hashlib.sha256((run_dir / 'final' / 'manifest.json').read_bytes()).hexdigest()}

    from tnlm.evaluate import evaluate_checkpoint
    evaluation = evaluate_checkpoint(config, result['evaluation_checkpoint'], tmp_path / 'evaluation',
        max_prediction_tokens=4, max_seconds=60, bootstrap_samples=10, device='cpu')
    assert evaluation['prediction_tokens'] == 4
    assert evaluation['billing']['status'] == 'complete'

    from tnlm.reporting import summarize_runs
    summary = summarize_runs([run_dir], tmp_path / 'summary')
    assert summary['runs'] == 1
    table = json.loads((tmp_path / 'summary' / 'comparison.json').read_text())
    assert table['runs'][0]['global_prediction_tokens'] == 7
    assert table['runs'][0]['optimizer'] == optimizer
    assert table['runs'][0]['configured_lr'] == config.train.lr
    assert table['runs'][0]['aux_adamw_lr'] == .002
