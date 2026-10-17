"""Distributed context forks are CPU/gloo only in this gate."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_context_fork_dry_run_and_flags_before_parent_io(config, tmp_path, monkeypatch):
    import tnlm.distributed_train as module
    config.raw['project']['dry_run'] = True
    config.raw['data']['context'] = 7
    monkeypatch.setattr(module, '_group', lambda: (0, 2))
    monkeypatch.setattr(module, '_phase', lambda label, callback: callback())
    monkeypatch.setattr(module, '_agree', lambda *args: None)
    result = module.distributed_train_run(config, tmp_path / 'dry', max_steps=1, max_seconds=60,
        resume=tmp_path / 'missing-parent', context_fork=True)
    assert result['dry_run'] and result['context_fork'] and result['context'] == 7
    assert result['parent_validation'] == 'pending'
    assert result['new_context'] == 7 and result['parent'] == str((tmp_path / 'missing-parent').resolve())
    for kwargs in [{'context_fork': True}, {'context_fork': 1, 'resume': tmp_path / 'missing-parent'}]:
        with pytest.raises(ValueError):
            module.distributed_train_run(config, tmp_path / 'bad', max_steps=1, max_seconds=60, **kwargs)
    assert not (tmp_path / 'dry').exists() and not (tmp_path / 'bad').exists()


@pytest.mark.slow
def test_two_rank_context_fork_exact_replay(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2',
        str(Path(__file__).parent / 'dist_workers' / 'context_replay.py'), str(tmp_path)],
        capture_output=True, text=True, timeout=360,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    (tmp_path / 'evidence.txt').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-10000:]
    for optimizer in ['adamw', 'muon']:
        assert optimizer + ': exact context-fork pending/replay/norm/RNG and prebilling rejection passed' in result.stdout
