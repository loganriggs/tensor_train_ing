"""CPU verification of the actual day-1 distributed smoke entrypoint."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/h100_trainer_smoke.py'


def test_checkpoint_tree_comparison_reports_errors():
    spec = importlib.util.spec_from_file_location('h100_trainer_smoke', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    left = {'weights': torch.tensor([1., 2.]), 'progress': {'steps': 4}}
    same = module.compare_tree(left, left)
    assert same['exact'] and same['max_abs_diff'] == 0.
    different = module.compare_tree(left, {'weights': torch.tensor([1., 2.25]), 'progress': {'steps': 4}})
    assert not different['exact'] and different['max_abs_diff'] == .25
    assert different['first_mismatches'] == ['root.weights']
    nonfinite = module.compare_tree([torch.tensor(float('nan')), torch.tensor(1.)],
                                    [torch.tensor(0.), torch.tensor(2.)])
    assert nonfinite['max_abs_diff'] is None and not nonfinite['exact']


@pytest.mark.slow
def test_smoke_runs_on_two_cpu_ranks(tmp_path):
    torchrun = shutil.which('torchrun') or str(Path(sys.executable).with_name('torchrun'))
    output = tmp_path/'smoke'
    command = [torchrun, '--standalone', '--nproc-per-node', '2', str(SCRIPT),
               '--output', str(output), '--steps', '4', '--backend', 'gloo', '--max-seconds', '120']
    result = subprocess.run(command, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout[-4000:]+result.stderr[-10000:]
    report = json.loads((output/'result.json').read_text())
    assert report['status'] == 'passed' and report['exact_resume']
    assert report['backend'] == 'gloo' and report['world_size'] == 2
    assert report['progress']['steps'] == 4
    assert all(value == 0 for value in report['max_abs_diffs'].values())
    assert set(report['max_abs_diffs']) == {'model', 'optimizer', 'sampler', 'progress', 'rng', 'metrics'}
    saved = (output/'result.json').read_bytes()
    rejected = subprocess.run(command, capture_output=True, text=True, timeout=90,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert rejected.returncode != 0
    assert (output/'result.json').read_bytes() == saved
