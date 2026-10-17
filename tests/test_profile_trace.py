"""Actual CPU Chrome traces, without invoking CUDA or training models."""
import hashlib
import json

import pytest
import torch

from tnlm.profiling import KernelProfile


def update(profiler, index):
    profiler.start_step(index)
    with profiler.region('tnlm/test-forward-and-loss'):
        parameter = torch.tensor([1., 2.], requires_grad=True)
        loss = parameter.square().sum()
    with profiler.region('tnlm/test-backward'):
        loss.backward()
    profiler.finish_step()


def test_actual_cpu_trace_named_regions_backward_warmup_and_bound(tmp_path):
    out = tmp_path / 'trace'
    profiler = KernelProfile(1, trace_dir=out, trace_steps=2, cuda=False, kernels=False)
    assert not out.exists()
    for index in range(5):
        update(profiler, index)
    profiler.close()
    report = profiler.report()
    assert report['trace_count'] == report['completed_steps'] == 2
    assert [item['step_index'] for item in report['traces']] == [1, 2]
    assert report['ranking'] == [] and report['total_cuda_time_us'] == 0
    assert report['capture_status'] == 'kernel_ranking_disabled'
    assert len(list(out.iterdir())) == 2
    for item in report['traces']:
        path = out / f'step-{item["step_index"]:06d}.json'
        payload = path.read_bytes()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
        names = [event.get('name', '') for event in json.loads(payload)['traceEvents']]
        assert 'tnlm/test-forward-and-loss' in names and 'tnlm/test-backward' in names
        assert any('Backward' in name or 'backward' in name for name in names)


def test_interrupted_cpu_trace_closes_once_without_export(tmp_path, monkeypatch):
    profiler = KernelProfile(0, trace_dir=tmp_path / 'traces', trace_steps=1, cuda=False, kernels=False)
    profiler.start_step(0)
    with profiler.region('interrupted'):
        torch.ones(3).sum()
    active = profiler.active
    original = active.__exit__
    exits = []
    def exit_once(*args):
        exits.append(True)
        return original(*args)
    monkeypatch.setattr(active, '__exit__', exit_once)
    profiler.close()
    profiler.close()
    assert exits == [True] and profiler.active is None
    assert profiler.report()['trace_count'] == profiler.report()['completed_steps'] == 0
    assert not (tmp_path / 'traces').exists()
    # An aborted capture does not consume the budget for a subsequent complete one.
    update(profiler, 1)
    assert profiler.report()['trace_count'] == 1


def test_no_overwrite_and_capture_released_on_export_failure(tmp_path):
    out = tmp_path / 'traces'
    out.mkdir()
    existing = out / 'step-000000.json'
    existing.write_text('precious')
    profiler = KernelProfile(0, trace_dir=out, trace_steps=1, cuda=False, kernels=False)
    with pytest.raises(FileExistsError):
        update(profiler, 0)
    assert profiler.active is None
    profiler.close()
    assert existing.read_text() == 'precious'
    assert list(out.iterdir()) == [existing]


@pytest.mark.parametrize('kwargs', [{'trace_steps': -1}, {'trace_steps': 101}, {'trace_steps': True},
    {'trace_steps': 1}, {'cuda': 1}, {'kernels': 'yes'}])
def test_invalid_controls_no_io(tmp_path, kwargs):
    with pytest.raises(ValueError):
        KernelProfile(0, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_no_capture_or_region_overhead_without_requested_capture(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('capture/range invoked while disabled')
    monkeypatch.setattr(torch.profiler, 'profile', forbidden)
    monkeypatch.setattr(torch.profiler, 'record_function', forbidden)
    profiler = KernelProfile(0, trace_dir=tmp_path / 'unused', cuda=False, kernels=False)
    update(profiler, 0)
    assert profiler.report()['completed_steps'] == 0
    assert not (tmp_path / 'unused').exists()


def test_trace_bound_preserves_cuda_ranking_across_later_updates_without_gpu(tmp_path, monkeypatch):
    import contextlib
    from types import SimpleNamespace
    captures, regions = [], []
    class Capture:
        def __enter__(self):
            self.exits = 0
        def __exit__(self, *args):
            self.exits += 1
        def events(self):
            return [SimpleNamespace(name='fake-CUDA', device_type=torch.autograd.DeviceType.CUDA,
                                    device_time_total=100.)]
        def export_chrome_trace(self, path):
            from pathlib import Path
            Path(path).write_text('{"traceEvents": []}')
    def factory(**kwargs):
        captures.append((kwargs['activities'], Capture()))
        return captures[-1][1]
    def region(name):
        regions.append(name)
        return contextlib.nullcontext()
    monkeypatch.setattr(torch.profiler, 'profile', factory)
    monkeypatch.setattr(torch.profiler, 'record_function', region)
    profiler = KernelProfile(1, trace_dir=tmp_path / 'traces', trace_steps=1)
    for index in range(4):
        profiler.start_step(index)
        with profiler.region('selected-only'):
            pass
        profiler.finish_step()
    assert captures[0][0] == [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    assert all(activities == [torch.profiler.ProfilerActivity.CUDA] for activities, _ in captures[1:])
    assert all(capture.exits == 1 for _, capture in captures)
    assert regions == ['selected-only']
    report = profiler.report()
    assert report['trace_count'] == 1 and report['completed_steps'] == 3
    assert report['total_cuda_time_us'] == 300. and report['ranking'][0]['calls'] == 3


def test_failed_export_never_publishes_partial_trace(tmp_path, monkeypatch):
    profiler = KernelProfile(0, trace_dir=tmp_path / 'traces', trace_steps=1, cuda=False, kernels=False)
    profiler.start_step(0)
    def fail(path):
        from pathlib import Path
        Path(path).write_text('partial trace')
        raise RuntimeError('injected exporter failure')
    monkeypatch.setattr(profiler.active, 'export_chrome_trace', fail)
    with pytest.raises(RuntimeError, match='exporter failure'):
        profiler.finish_step()
    assert profiler.active is None
    profiler.close()
    assert list((tmp_path / 'traces').iterdir()) == []
    assert profiler.report()['trace_count'] == profiler.report()['completed_steps'] == 0
