"""Distributed TaperNorm trainer lifecycle on two CPU ranks."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_two_rank_taper_lifecycle_and_exact_resume(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    worker = Path(__file__).parent / 'dist_workers' / 'taper_train.py'
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2', str(worker), str(tmp_path)],
        capture_output=True, text=True, timeout=360,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    (tmp_path/'evidence.txt').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-12000:]
    assert 'Taper distributed freeze, final polynomial gate, exact resume, and cooldown passed' in result.stdout
