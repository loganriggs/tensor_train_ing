"""Collective early-stop FSDP calibration uses only tiny CPU/gloo models."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_two_rank_sharded_lagged_calibration_replay_and_collective_failures(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    if not Path(torchrun).exists():
        pytest.skip('torchrun unavailable')
    worker = Path(__file__).parent / 'dist_workers' / 'calibration_replay.py'
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2', str(worker), str(tmp_path)],
        capture_output=True, text=True, timeout=240,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    (tmp_path / 'calibration-evidence.txt').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-9000:]
    assert '17-site deep overflow calibration: exact state, unchanged weights, full forward/backward/update passed' in result.stdout
    for failure in ('rank_deadline', 'invalid_token', 'batch_mismatch'):
        assert f'{failure}: rejected before offending forward, state rollback, mode/hooks/stat cleanup and retry passed' in result.stdout
