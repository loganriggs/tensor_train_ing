"""End-to-end distributed trainer integration runs exclusively CPU/gloo."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tnlm.distributed_train import distributed_train_run


def test_distributed_trainer_requires_initialized_group(config, tmp_path):
    with pytest.raises(ValueError, match='initialized process group'):
        distributed_train_run(config, tmp_path / 'no-group', max_steps=1, max_seconds=1)
    assert not (tmp_path / 'no-group').exists()


@pytest.mark.slow
def test_two_rank_cpu_trainer_global_objective_and_replay(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    if not Path(torchrun).exists():
        pytest.skip('torchrun unavailable')
    worker = Path(__file__).parent / 'dist_workers' / 'train_replay.py'
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2', str(worker), str(tmp_path)],
        capture_output=True, text=True, timeout=360,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    (tmp_path / 'numeric-evidence.txt').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-9000:]
    for package in ('pre_rms', 'spectral_rms', 'lagged', 'pre_rms_fp32'):
        assert f'{package}: serial global normalization/clipping and exact distributed resume passed' in result.stdout
    assert 'single instance ledger and cooldown replay passed' in result.stdout
    assert 'collective retention admission rejects before checkpoint writes' in result.stdout
