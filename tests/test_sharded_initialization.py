"""CPU/gloo replay gate for meta-sharded restore destination allocation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_meta_sharded_restore_exact_state_forward_and_update(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    if not Path(torchrun).exists():
        pytest.skip('torchrun unavailable')
    worker = Path(__file__).parent / 'dist_workers' / 'initialization_replay.py'
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2', str(worker), str(tmp_path)],
                            capture_output=True, text=True, timeout=240,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-7000:]
    for case in ('pre_rms', 'spectral_rms', 'lagged', 'tied'):
        assert f'{case}: meta-only allocation and exact state/forward/update replay' in result.stdout
