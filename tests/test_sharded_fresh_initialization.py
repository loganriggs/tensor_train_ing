"""Fresh canonical initialization must survive real two-rank sharding."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_fresh_sharded_initialization_matches_serial_state_rng_and_logits(tmp_path):
    worker = Path(__file__).parent / 'dist_workers' / 'fresh_initialization.py'
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                             '--nproc-per-node', '2', str(worker)], capture_output=True,
                            text=True, timeout=120,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-7000:]
    assert 'canonical fresh sharding: all cases passed' in result.stdout
