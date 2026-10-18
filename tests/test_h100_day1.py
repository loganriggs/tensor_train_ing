"""Day-one orchestration policy tested without GPUs or external processes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def day1(monkeypatch):
    spec = importlib.util.spec_from_file_location('h100_day1_test_target', ROOT / 'scripts/h100_day1.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def arguments(day1, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    validation = tmp_path / 'validation.json'
    validation.write_text('{}')
    return ['--output', str(tmp_path / 'day1'), '--manifest', str(manifest),
            '--validation-manifest', str(validation), '--gpu-hourly-rate-usd', '2.5',
            '--peak-tflops-per-gpu', '989', '--budget-usd', '3', '--max-seconds', '3600']


def _probe(path):
    report = {'gpu_count': 8, 'cuda_available': True, 'nccl_available': True,
              'gpus': [{'name': 'mock H100', 'memory_total_bytes': 80 * 1024**3} for _ in range(8)],
              'torch_version': 'mock', 'triton_version': 'mock', 'driver_version': 'mock',
              'cuda_version': 'mock', 'free_disk_bytes': 1024**4, 'free_ram_bytes': 1024**4}
    Path(path).write_text(json.dumps(report))
    return report


def _stage(command):
    joined = ' '.join(map(str, command))
    if 'fsdp_equivalence.py' in joined:
        return 'equivalence'
    if 'h100_trainer_smoke.py' in joined:
        return 'smoke-' + str(command[command.index('--nproc-per-node') + 1])
    if 'h100_canary.py' in joined:
        return 'canary'
    if 'tnlm.kernels.tune' in joined:
        return 'tune'
    if 'tests/test_kernels.py' in command:
        return 'gpu-tests'
    return 'cpu-tests'


class FakeRunner:
    def __init__(self, *, fail=None, tuning_ms=1.25):
        self.calls = []
        self.fail = fail
        self.tuning_ms = tuning_ms

    def __call__(self, command, *, cwd, env, log_path, timeout):
        command = list(map(str, command))
        stage = _stage(command)
        self.calls.append({'stage': stage, 'command': command, 'cwd': cwd, 'env': dict(env), 'timeout': timeout})
        stdout = ''
        if stage == 'equivalence':
            world = command[command.index('--nproc-per-node') + 1]
            stdout = '\n'.join(f'{name}: world={world} backend=nccl max|param diff| = 0.000e+00'
                               for name in ('pre_rms', 'spectral_rms', 'lagged'))
        elif stage == 'tune':
            stdout = json.dumps({'gpu': 'mock H100', 'head_dim': 128,
                                 'best': {'ms': self.tuning_ms, 'block': 64}})
        elif stage.startswith('smoke-') or stage == 'canary':
            output = Path(command[command.index('--output') + 1])
            output.mkdir(parents=True, exist_ok=False)
            (output / 'result.json').write_text(json.dumps({'status': 'passed', 'mock': True}))
        code = 7 if stage == self.fail else 0
        Path(log_path).write_text(stdout + ('\ninjected failure' if code else ''))
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr='')


def test_day1_stage_order_and_isolated_canary_configuration(day1, arguments, tmp_path):
    runner = FakeRunner()
    pilot = ROOT / 'configs/pilot_1b.yaml'
    pilot_before = pilot.read_bytes()
    report = day1.run(day1.parse_args(arguments), runner=runner, probe=_probe)
    assert report['status'] == 'passed'
    assert report['acceptance_complete'] is True
    assert [call['stage'] for call in runner.calls] == [
        'cpu-tests', 'gpu-tests', 'equivalence', 'smoke-2', 'smoke-8', 'tune', 'canary']
    output = tmp_path / 'day1'
    persisted = json.loads((output / 'report.json').read_text())
    assert persisted['status'] == 'passed'
    assert (output / 'report.md').is_file()
    assert pilot.read_bytes() == pilot_before
    config = yaml.safe_load((output / 'canary.yaml').read_text())
    assert config['project']['dry_run'] is False
    assert Path(config['project']['spend_ledger']).resolve() == output / 'canary-spend.json'
    assert config['project']['gpu_hourly_rate_usd'] == 2.5
    assert config['project']['reserve_usd'] == 0
    assert config['project']['total_budget_usd'] == 3
    assert config['project']['save_initial_checkpoint'] is False
    assert config['project']['keep_recovery_checkpoints'] == 0
    assert config['project']['milestone_tokens'] == []
    assert config['train']['compile'] is False
    assert config['train']['qk_clip_tau'] == 0
    assert config['train']['qk_clip_source'] == 'batch_bound'
    assert config['train']['rung_total_tokens'] == 200 * config['train']['global_prediction_tokens_per_update']
    assert Path(config['data']['manifest']).resolve() == tmp_path / 'manifest.json'
    assert Path(config['evaluation']['validation_manifest']).resolve() == tmp_path / 'validation.json'
    canary = runner.calls[-1]['command']
    assert canary[canary.index('--steps') + 1] == '200'
    assert canary[canary.index('--nproc-per-node') + 1] == '8'
    assert runner.calls[0]['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert '-n' in runner.calls[0]['command'] and 'auto' in runner.calls[0]['command']


def test_day1_skip_flags_preserve_remaining_order(day1, arguments):
    runner = FakeRunner()
    args = day1.parse_args(arguments + ['--skip-cpu-tests', '--skip-equivalence', '--skip-smoke', '--skip-canary'])
    report = day1.run(args, runner=runner, probe=_probe)
    assert report['status'] == 'passed'
    assert [call['stage'] for call in runner.calls] == ['gpu-tests', 'tune']
    skipped = {row['name'] for row in report['stages'] if row['status'] == 'skipped'}
    assert {'cpu-tests', 'equivalence', 'smoke-2', 'canary'} <= skipped


def test_day1_all_stages_skippable(day1, arguments):
    runner = FakeRunner()
    flags = ['--skip-' + name for name in ('env', 'cpu-tests', 'gpu-tests', 'equivalence', 'smoke', 'tune', 'canary')]
    report = day1.run(day1.parse_args(arguments + flags), runner=runner, probe=_probe)
    assert report['status'] == 'passed'
    assert runner.calls == []
    assert report['acceptance_complete'] is False
    assert all(row['status'] == 'skipped' for row in report['stages'])


def test_day1_stops_and_persists_first_failure(day1, arguments, tmp_path):
    runner = FakeRunner(fail='gpu-tests')
    report = day1.run(day1.parse_args(arguments), runner=runner, probe=_probe)
    assert report['status'] == 'failed'
    assert [call['stage'] for call in runner.calls] == ['cpu-tests', 'gpu-tests']
    assert report['error']
    saved = json.loads((tmp_path / 'day1/report.json').read_text())
    failed = [row for row in saved['stages'] if row['status'] == 'failed']
    assert len(failed) == 1 and failed[0]['name'] == 'gpu-tests'
    assert not (tmp_path / 'day1/canary.yaml').exists()


def test_day1_refuses_existing_output_without_mutation(day1, arguments, tmp_path):
    output = tmp_path / 'day1'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('existing results')
    runner = FakeRunner()
    with pytest.raises(FileExistsError):
        day1.run(day1.parse_args(arguments), runner=runner, probe=_probe)
    assert sentinel.read_text() == 'existing results'
    assert list(output.iterdir()) == [sentinel]
    assert runner.calls == []


@pytest.mark.parametrize('milliseconds', [float('nan'), float('inf'), -1., 0.])
def test_day1_rejects_invalid_tuning_even_with_zero_exit_status(day1, arguments, milliseconds):
    runner = FakeRunner(tuning_ms=milliseconds)
    flags = ['--skip-' + name for name in ('cpu-tests', 'gpu-tests', 'equivalence', 'smoke')]
    report = day1.run(day1.parse_args(arguments + flags), runner=runner, probe=_probe)
    assert report['status'] == 'failed'
    assert [call['stage'] for call in runner.calls] == ['tune']
    assert 'tune' in report['error'].lower() or 'finite' in report['error'].lower()


@pytest.mark.parametrize('budget', [3., .5])
def test_canary_launch_budget_passes_actual_ledger_reservation(day1, arguments, tmp_path, budget):
    from tnlm.budget import SpendLedger
    arguments[arguments.index('--budget-usd') + 1] = str(budget)
    flags = ['--skip-' + name for name in ('cpu-tests', 'gpu-tests', 'equivalence', 'smoke', 'tune')]
    runner = FakeRunner()
    report = day1.run(day1.parse_args(arguments + flags), runner=runner, probe=_probe)
    assert report['status'] == 'passed'
    assert report['acceptance_complete'] is False
    command = runner.calls[-1]['command']
    project = yaml.safe_load((tmp_path / 'day1/canary.yaml').read_text())['project']
    ledger = SpendLedger(tmp_path / 'reservation-check.json',
                         total_budget=project['total_budget_usd'], reserve=project['reserve_usd'],
                         stage_budgets=project.get('stage_budgets_usd'))
    amount = ledger.begin('mock-canary', project['gpu_hourly_rate_usd'],
                          float(command[command.index('--max-seconds') + 1]),
                          run_budget=float(command[command.index('--run-budget-usd') + 1]),
                          cost_margin_fraction=project.get('cost_margin_fraction', .1), stage='D')
    assert 0 < amount <= budget
