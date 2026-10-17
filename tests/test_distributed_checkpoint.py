"""Distributed recovery primitive: CPU only, subprocess timeout and bounded group timeout."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from tnlm.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint


def test_distributed_checkpoint_requires_initialized_group(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match='initialized process group'):
        save_distributed_checkpoint(tmp_path / 'saved', model, optimizer, None,
                                    config={}, progress={}, manifest_hash='abc')
    with pytest.raises(ValueError, match='initialized process group'):
        load_distributed_checkpoint(tmp_path / 'saved', model, optimizer, None)


@pytest.mark.slow
def test_two_rank_cpu_fsdp_checkpoint_exact_replay(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    if not Path(torchrun).exists():
        pytest.skip('torchrun unavailable')
    worker = Path(__file__).parent / 'dist_workers' / 'checkpoint_replay.py'
    result = subprocess.run([torchrun, '--standalone', '--nproc-per-node', '2', str(worker), str(tmp_path)],
        capture_output=True, text=True, timeout=240,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-7000:]
    assert 'buffer identity mismatch: model/optimizer/sampler/RNG unchanged before installation' in result.stdout
    for package in ('pre_rms', 'spectral_rms', 'lagged'):
        assert f'{package}: world=2 exact next-batch/RNG/optimizer/update replay' in result.stdout
        assert f'{package}: FP32 trained/exported actual analysis and frozen state/mode/hooks passed' in result.stdout
