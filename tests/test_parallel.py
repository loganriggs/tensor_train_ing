"""FSDP2 data-parallel equivalence: 2 ranks (gloo/CPU) vs one process on the same global batch."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).parent / "dist_workers" / "fsdp_equivalence.py"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("torchrun") is None and not Path(sys.executable).with_name("torchrun").exists(),
                    reason="torchrun unavailable")
def test_fsdp2_two_rank_step_matches_single_process():
    torchrun = shutil.which("torchrun") or str(Path(sys.executable).with_name("torchrun"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # force gloo/CPU so it runs on any machine
    r = subprocess.run([torchrun, "--nproc-per-node", "2", "--master-port", "29517", str(WORKER)],
                       capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for pkg in ("pre_rms", "spectral_rms", "lagged"):
        assert f"{pkg}: world=2" in r.stdout
