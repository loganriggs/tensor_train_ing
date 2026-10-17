"""CPU/gloo cosine integration; no GPU execution."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

@pytest.mark.slow
def test_two_rank_cosine_replay(tmp_path):
    torchrun=shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    result=subprocess.run([torchrun,'--standalone','--nproc-per-node','2',
        str(Path(__file__).parent/'dist_workers/cosine_replay.py'),str(tmp_path)],
        capture_output=True,text=True,timeout=360,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'))
    (tmp_path/'evidence.txt').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout[-5000:]+result.stderr[-10000:]
    assert 'cosine exact warmup-boundary replay and prebilling rejection passed' in result.stdout
