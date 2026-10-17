import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_two_rank_muon_training_and_cooldown_replay(tmp_path):
    worker = Path(__file__).parent / 'dist_workers' / 'muon_train_replay.py'
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node', '2', str(worker), str(tmp_path)], capture_output=True, text=True,
        timeout=240, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    (tmp_path / 'muon-evidence.txt').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    for norm in ('pre_rms', 'spectral_rms', 'lagged'):
        assert f'{norm}: Muon training and cooldown exact state/data/RNG replay passed' in result.stdout
