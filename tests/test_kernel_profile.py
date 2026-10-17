"""CPU-only tests of CUDA event accounting and complete-update capture hooks."""
import importlib
import json
from types import SimpleNamespace
import pytest
import torch
from tnlm.profiling import KernelProfile
from tnlm.train import train_run, profile_run


def event(name, duration, cuda=True, annotation=False):
    device_type = (
        torch.autograd.DeviceType.CUDA if cuda else torch.autograd.DeviceType.CPU
    )
    return SimpleNamespace(
        name=name, device_time_total=duration, device_type=device_type,
        is_user_annotation=annotation
    )


def fake_factory(monkeypatch, events):
    captures = []

    class Capture:

        def __enter__(self):
            self.entered = True
            self.exits = 0
            return self

        def __exit__(self, *args):
            self.exits += 1

        def events(self):
            return events

    def factory(**kwargs):
        assert kwargs['activities'] == [torch.profiler.ProfilerActivity.CUDA]
        capture = Capture()
        captures.append(capture)
        return capture
    monkeypatch.setattr(torch.profiler, 'profile', factory)
    return captures


def test_raw_events_ignore_cpu_aggregates_and_denominators(monkeypatch):
    captures = fake_factory(
        monkeypatch,
        [event('cpu-parent', 9000, False),
         event('tnlm.forward', 9000, annotation=True),
         event('Optimizer.step#AdamW.step', 5000, annotation=True), event('A', 1000),
         event('A', 2000), event('B', 1000)],
    )
    p = KernelProfile(2)
    for i in range(4):
        p.start_step(i)
        p.finish_step()
    assert len(captures) == 2 and all((c.exits == 1 for c in captures))
    report = p.report()
    assert report['completed_steps'] == 2 and report['total_cuda_time_us'] == 8000
    a, b = report['ranking']
    assert [a['name'], b['name']] == ['A', 'B']
    assert a['calls'] == 4 and a['mean_cuda_us_per_call'] == 1500
    assert a['mean_cuda_ms_per_step'] == 3 and a['fraction_of_device_time'] == 0.75
    assert b['mean_cuda_ms_per_step'] == 1


def test_interrupted_capture_cleanup_discards_update(monkeypatch):
    captures = fake_factory(monkeypatch, [event('A', 100)])
    p = KernelProfile(0)
    p.start_step(0)
    p.close()
    p.close()
    assert captures[0].exits == 1
    assert p.report()['completed_steps'] == 0 and p.report()['ranking'] == []


@pytest.mark.parametrize('duration', [float('nan'), float('inf'), -1])
def test_invalid_device_duration_rejected(monkeypatch, duration):
    captures = fake_factory(monkeypatch, [event('A', duration)])
    p = KernelProfile(0)
    p.start_step(0)
    with pytest.raises(ValueError, match='duration'):
        p.finish_step()
    p.close()
    assert captures[0].exits == 1


class Observer:

    def __init__(self, warmup=1):
        self.warmup = warmup
        self.active = False
        self.starts = []
        self.finishes = 0
        self.closed = 0
        self.forwards = 0

    def start_step(self, index):
        self.starts.append(index)
        self.active = index >= self.warmup

    def finish_step(self):
        if self.active:
            self.finishes += 1
        self.active = False

    def close(self):
        self.active = False
        self.closed += 1

    def report(self):
        return {'completed_steps': self.finishes}


def test_actual_training_hook_accumulation_saves_and_updates(config, tmp_path, monkeypatch):
    observer = Observer()
    counts = {'updates': 0}
    original_step = torch.optim.AdamW.step

    def step(self, *args, **kwargs):
        counts['updates'] += 1
        return original_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    from tnlm.model import TensorLM
    forward = TensorLM.forward

    def observed_forward(self, *args, **kwargs):
        if observer.active:
            observer.forwards += 1
        return forward(self, *args, **kwargs)
    monkeypatch.setattr(TensorLM, 'forward', observed_forward)
    checkpoint = importlib.import_module('tnlm.checkpoint')
    save = checkpoint.save_checkpoint

    def observed_save(*args, **kwargs):
        assert not observer.active
        return save(*args, **kwargs)
    monkeypatch.setattr(checkpoint, 'save_checkpoint', observed_save)
    config.raw['project'].update(checkpoint_every_steps=1, milestone_tokens=[])
    result = train_run(config, tmp_path / 'run', max_steps=3, max_seconds=60, device='cpu', _step_profiler=observer)
    assert counts['updates'] == 3 and result['progress']['steps'] == 3
    assert observer.starts == [0, 1, 2] and observer.finishes == 2 and (observer.closed == 1)
    assert observer.forwards >= 4
    assert result['kernel_profile']['completed_steps'] == 2


def test_interrupted_training_closes_capture_and_finalizes_bill(config, tmp_path, monkeypatch):
    observer = Observer(0)
    from tnlm.model import TensorLM

    def failed(*args, **kwargs):
        raise RuntimeError('injected forward failure')
    monkeypatch.setattr(TensorLM, 'forward', failed)
    with pytest.raises(RuntimeError, match='injected'):
        train_run(config, tmp_path / 'failed', max_steps=1, max_seconds=60, device='cpu', _step_profiler=observer)
    assert observer.closed == 1 and observer.finishes == 0 and (not observer.active)
    bill = json.loads((tmp_path / 'spend.json').read_text())['runs'][str((tmp_path / 'failed').resolve())]
    assert bill['status'] == 'failed' and bill['reserved_usd'] == 0


def test_cpu_kernel_profile_rejected_before_output_or_billing(config, tmp_path):
    with pytest.raises(ValueError, match='CUDA'):
        profile_run(config, tmp_path / 'profile', steps=1, max_seconds=60, device='cpu', kernels=True)
    assert not (tmp_path / 'profile').exists() and (not (tmp_path / 'spend.json').exists())



def test_cli_forwards_kernel_profile_controls(config, tmp_path, monkeypatch, capsys):
    import yaml
    from tnlm import cli

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    captured = {}

    def fake_profile(cfg, run_dir, **kwargs):
        captured.update(kwargs)
        captured['run_dir'] = run_dir
        return {'command': 'profile', 'kernels': kwargs['kernels']}

    monkeypatch.setattr(cli, 'profile_run', fake_profile)
    cli.main([
        'profile', '--config', str(config_path),
        '--run-dir', str(tmp_path / 'profile'),
        '--steps', '3', '--warmup-steps', '2',
        '--max-seconds', '20', '--device', 'cuda', '--kernels',
    ])
    assert captured == {
        'steps': 3,
        'warmup_steps': 2,
        'max_seconds': 20.0,
        'device': 'cuda',
        'run_budget_usd': None,
        'kernels': True,
        'run_dir': str(tmp_path / 'profile'),
    }
    assert json.loads(capsys.readouterr().out)['kernels'] is True
    assert not (tmp_path / 'profile').exists()



def test_cuda_dry_run_preserves_kernel_flag_without_capture(config, tmp_path, monkeypatch):
    config.raw['project']['dry_run'] = True

    def forbidden_capture(*args, **kwargs):
        raise AssertionError('dry run started a CUDA profiler')

    monkeypatch.setattr(torch.profiler, 'profile', forbidden_capture)
    result = profile_run(
        config, tmp_path / 'profile', steps=3, warmup_steps=2,
        max_seconds=60, device='cuda', kernels=True,
    )
    assert result['dry_run'] is True
    assert result['kernels'] is True
    assert result['timed_steps'] == 3
    assert result['warmup_steps'] == 2
    assert not (tmp_path / 'profile').exists()
    assert not (tmp_path / 'spend.json').exists()
