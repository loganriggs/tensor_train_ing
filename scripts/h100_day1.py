"""Auditable, fail-fast single-node multi-GPU acceptance command."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.report-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def probe_environment(output):
    import torch
    try:
        triton = importlib.metadata.version('triton')
    except importlib.metadata.PackageNotFoundError:
        triton = None
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=20, check=False)
        driver_versions, driver_error = driver.stdout.splitlines(), driver.stderr if driver.returncode else None
    except (OSError, subprocess.TimeoutExpired) as error:
        driver_versions, driver_error = [], str(error)
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            memory[key] = int(value.split()[0]) * 1024
    return {'gpu_count': torch.cuda.device_count(), 'cuda_available': torch.cuda.is_available(),
            'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                      'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())],
            'driver_versions': driver_versions, 'driver_probe_error': driver_error, 'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__, 'triton_version': triton,
            'nccl_available': torch.distributed.is_nccl_available(),
            'free_disk_bytes': shutil.disk_usage(output.parent).free, 'ram_bytes': memory,
            'python': sys.version, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def run_command(command, *, cwd, env, log_path, timeout):
    """Keep logs on disk and clean up the whole stage process group on interruption."""
    with log_path.open('w') as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                # The leader may have exited while an ignoring grandchild is
                # still alive. Always finish cleaning the isolated group.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
    return subprocess.CompletedProcess(command, code, stdout=log_path.read_text())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--validation-manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/pilot_1b.yaml')
    p.add_argument('--gpu-hourly-rate-usd', type=float, required=True,
                   help='Actual WHOLE NODE hourly price, not price per GPU')
    p.add_argument('--peak-tflops-per-gpu', type=float, required=True,
                   help='Dense peak FLOPs for the compute precision used, per GPU')
    p.add_argument('--budget-usd', type=float, default=100.)
    p.add_argument('--max-seconds', type=float, default=3600., help='Maximum canary training seconds')
    p.add_argument('--timeout', type=float, default=14400., help='Whole runbook watchdog seconds, including tests')
    p.add_argument('--steps', type=int, default=200)
    for stage in ('env', 'cpu-tests', 'gpu-tests', 'equivalence', 'smoke', 'tune', 'canary'):
        p.add_argument('--skip-' + stage, action='store_true')
    return p.parse_args(argv)


def parse_equivalence(text, world):
    matches = re.findall(r'(\w+): world=(\d+) backend=(\w+) max\|param diff\| = ([^\s]+)', text)
    rows = {package: {'world': int(n), 'backend': backend, 'max_parameter_diff': float(diff)}
            for package, n, backend, diff in matches}
    for package in ('pre_rms', 'spectral_rms', 'lagged'):
        row = rows.get(package)
        if (row is None or row['world'] != world or row['backend'] != 'nccl'
                or not math.isfinite(row['max_parameter_diff']) or not 0 <= row['max_parameter_diff'] < 1e-10):
            raise ValueError('NCCL equivalence missing or failed for ' + package)
    return rows


def parse_tuning(text):
    decoder = json.JSONDecoder()
    result = None
    for match in re.finditer(r'\{', text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start():])
            if isinstance(candidate, dict) and 'best' in candidate:
                result = candidate
                break
        except ValueError:
            continue
    if result is None:
        raise ValueError('tuning output contains no result')
    ms = result.get('best', {}).get('ms')
    if type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0:
        raise ValueError('tuning found no finite positive best time')
    # Failed alternatives may use Infinity in the tuner stdout; preserve the
    # raw log, but serialize unavailable measurements as null in strict JSON.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value
    return finite(result)


def run(args, *, runner=run_command, probe=probe_environment):
    for name in ('gpu_hourly_rate_usd', 'peak_tflops_per_gpu', 'budget_usd', 'max_seconds', 'timeout'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if args.steps <= 0:
        raise ValueError('steps must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    started = time.monotonic()
    deadline = started + args.timeout
    report = {'status': 'running', 'output': str(output), 'stages': [],
              'whole_node_hourly_rate_usd': args.gpu_hourly_rate_usd,
              'budget_usd': args.budget_usd, 'peak_tflops_per_gpu': args.peak_tflops_per_gpu,
              'limitations': ['CPU tests hide CUDA; NCCL/CUDA checks require the rental node.',
                'Canary disables active QK clipping and compilation: distributed support is unavailable.',
                'Tile tuning is reported, not installed into the production kernel table.',
                'MFU uses the supplied dense peak and the approximate 6*N*tokens FLOP model.',
                'budget-usd caps the canary ledger only; tests, tuning and smoke rental time are additional.',
                'max-seconds caps canary training; timeout is the separate whole-command watchdog.',
                'The retained constant schedule can create mandatory 90% branch and final checkpoints.']}
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['estimated_node_cost_usd'] = report['elapsed_seconds'] * args.gpu_hourly_rate_usd / 3600
        atomic_json(output / 'report.json', report)
        lines = ['# H100 day-1 report', '', 'Status: ' + report['status'], '',
                 'Only requested stages are covered; skipped stages are not acceptance evidence.', '']
        lines.extend('- ' + row['name'] + ': ' + row['status'] for row in report['stages'])
        if 'error' in report:
            lines.extend(['', report['error']])
        lines.extend(['', 'Limitations:', ''])
        lines.extend('- ' + item for item in report['limitations'])
        temporary = output / '.report.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        os.replace(temporary, output / 'report.md')
    def stage(name, skipped, action):
        row = {'name': name, 'status': 'skipped' if skipped else 'running'}
        report['stages'].append(row)
        save()
        if skipped:
            return
        before = time.monotonic()
        try:
            row['result'] = action(row)
            row['status'] = 'passed'
        except BaseException as error:
            row['status'], row['error'] = 'failed', str(error)
            raise
        finally:
            row['seconds'] = time.monotonic() - before
            save()
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('day-1 time or node-cost budget exhausted')
        return seconds
    def execute(row, command, *, cpu=False):
        env = dict(os.environ)
        env.setdefault('OMP_NUM_THREADS', '1')
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        log = output / 'logs' / (row['name'] + '.log')
        row.update(command=[str(arg) for arg in command], log=str(log))
        save()
        result = runner(row['command'], cwd=ROOT, env=env, log_path=log, timeout=remaining())
        row['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f"{row['name']} exited {result.returncode}; see {log}")
        return result.stdout or ''
    def worker_result(path):
        result = json.loads((path / 'result.json').read_text())
        if result.get('status') != 'passed':
            raise ValueError('worker did not report passed: ' + str(path))
        return result
    python = sys.executable
    torchrun = lambda n: [python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', str(n)]
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise RuntimeError('day-1 interrupted by SIGTERM')
    signal.signal(signal.SIGTERM, terminated)
    try:
        environment = {}
        def environment_stage(row):
            environment.update(probe(output / 'environment.json'))
            atomic_json(output / 'environment.json', environment)
            return environment
        stage('env', args.skip_env, environment_stage)
        need_gpu = not all((args.skip_gpu_tests, args.skip_equivalence, args.skip_smoke, args.skip_tune, args.skip_canary))
        def gpu_preflight(row):
            if not environment:
                environment.update(probe(output / 'environment.json'))
                report['required_gpu_preflight'] = environment
            if not environment.get('cuda_available') or not environment.get('nccl_available') or environment.get('gpu_count', 0) < 2:
                raise ValueError('GPU stages require at least two visible CUDA GPUs and NCCL')
            return {'gpu_count': environment['gpu_count'], 'nccl_available': True}
        stage('gpu-preflight', not need_gpu, gpu_preflight)
        count = environment.get('gpu_count', 0)
        stage('cpu-tests', args.skip_cpu_tests, lambda row: {'stdout_tail': execute(row,
            [python, '-m', 'pytest', '-n', 'auto', '-m', 'not slow', '--ignore=tests/test_kernels.py', '--basetemp', str(output / 'pytest-cpu')], cpu=True)[-4000:]})
        stage('gpu-tests', args.skip_gpu_tests, lambda row: {'stdout_tail': execute(row,
            [python, '-m', 'pytest', 'tests/test_kernels.py', '--basetemp', str(output / 'pytest-gpu')])[-4000:]})
        stage('equivalence', args.skip_equivalence, lambda row: parse_equivalence(execute(row,
            torchrun(count) + [str(ROOT / 'tests/dist_workers/fsdp_equivalence.py')]), count))
        for n in dict.fromkeys((2, count)):
            if n < 2:
                continue
            destination = output / f'smoke-{n}'
            def smoke(row, n=n, destination=destination):
                execute(row, torchrun(n) + [str(ROOT / 'scripts/h100_trainer_smoke.py'), '--output', str(destination),
                    '--steps', '20', '--backend', 'nccl', '--hourly-rate', str(args.gpu_hourly_rate_usd),
                    '--max-seconds', str(min(600., remaining()))])
                return worker_result(destination)
            stage(f'smoke-{n}', args.skip_smoke, smoke)
        stage('tune', args.skip_tune, lambda row: parse_tuning(execute(row,
            [python, '-m', 'tnlm.kernels.tune', '--head-dim', '128'])))
        def canary(row):
            for name in ('manifest', 'validation_manifest', 'config'):
                if not getattr(args, name).is_file():
                    raise ValueError(name + ' must name an existing file')
            cfg = yaml.safe_load(args.config.read_text())
            project = cfg.setdefault('project', {})
            project.update(dry_run=False, total_budget_usd=args.budget_usd, reserve_usd=0,
                spend_ledger=str(output / 'canary-spend.json'), gpu_hourly_rate_usd=args.gpu_hourly_rate_usd,
                save_initial_checkpoint=False, keep_recovery_checkpoints=0, milestone_tokens=[],
                stage='D', stage_budgets={'D': args.budget_usd}, cost_margin_fraction=.1)
            cfg.setdefault('data', {})['manifest'] = str(args.manifest.resolve())
            cfg.setdefault('evaluation', {})['validation_manifest'] = str(args.validation_manifest.resolve())
            train = cfg.setdefault('train', {})
            batch = train['global_prediction_tokens_per_update']
            if type(batch) is not int or batch <= 0:
                raise ValueError('pilot global batch must be a positive integer')
            train.update(compile=False, qk_clip_tau=0., qk_clip_source='batch_bound',
                         rung_total_tokens=args.steps * batch)
            config_path = output / 'canary.yaml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            report['canary_config'] = str(config_path)
            report['canary_overrides'] = {'project': project, 'train': train,
                'data_manifest': cfg['data']['manifest'], 'validation_manifest': cfg['evaluation']['validation_manifest']}
            destination = output / 'canary'
            seconds = min(args.max_seconds, remaining(),
                          .999 * args.budget_usd * 3600 / args.gpu_hourly_rate_usd / 1.1)
            budget = args.budget_usd
            execute(row, torchrun(count) + [str(ROOT / 'scripts/h100_canary.py'), '--config', str(config_path),
                '--output', str(destination), '--steps', str(args.steps), '--max-seconds', str(seconds),
                '--peak-tflops-per-gpu', str(args.peak_tflops_per_gpu), '--run-budget-usd', str(budget)])
            return worker_result(destination)
        stage('canary', args.skip_canary, canary)
        report['status'] = 'passed'
        report['acceptance_complete'] = all(row['status'] == 'passed' for row in report['stages'])
    except BaseException as error:
        report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        save()
    return report


def main(argv=None):
    report = run(parse_args(argv))
    print(json.dumps({'status': report['status'], 'report': str(Path(report['output']) / 'report.json')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
