"""CPU-only logical-matrix Muon and distributed composite-optimizer recovery."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_two_rank_muon_global_matrices_and_exact_checkpoint_replay(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    if not Path(torchrun).exists():
        pytest.skip('torchrun unavailable')
    worker = Path(__file__).parent / 'dist_workers' / 'muon_replay.py'
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2', str(worker), str(tmp_path)],
                            capture_output=True, text=True, timeout=300,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-8000:]
    evidence = json.loads((tmp_path / 'muon-semantics.json').read_text())
    assert evidence['serial_weight_atol'] == 4e-4 and evidence['serial_weight_rtol'] == 0
    assert len(evidence['measurements']) == 3
    for row in evidence['measurements']:
        assert row['dtensor_vs_serial_max_abs'] <= 4e-4
        assert row['momentum_vs_serial_max_abs'] == 0
        assert row['naive_local_vs_serial_max_abs'] > 4e-4
        assert row['naive_local_vs_dtensor_max_abs'] > 4e-4
    for shape in ('8x4', '4x8', '7x5'):
        assert f'{shape}: logical global-matrix Muon within fixed BF16 tolerance' in result.stdout
    for package in ('pre_rms', 'spectral_rms', 'lagged'):
        assert f'{package}: initial state and exact Muon/AdamW checkpoint replay' in result.stdout
