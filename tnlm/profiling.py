"""Bounded CUDA device-event accounting for complete training updates."""
from collections import defaultdict
import math
import contextlib
import hashlib
import os
from pathlib import Path
import tempfile

import torch


class KernelProfile:
    def __init__(self, warmup_steps, *, trace_dir=None, trace_steps=0, cuda=True, kernels=True):
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError('warmup_steps must be a nonnegative integer')
        if type(trace_steps) is not int or not 0 <= trace_steps <= 100:
            raise ValueError('trace_steps must be an integer in [0,100]')
        if trace_steps and trace_dir is None:
            raise ValueError('positive trace_steps requires trace_dir')
        if type(cuda) is not bool or type(kernels) is not bool:
            raise ValueError('cuda/kernels must be Boolean')
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.trace_steps, self.cuda, self.kernels = trace_steps, cuda, kernels
        self.traces = []
        self.active_trace = False
        self.active_index = None
        self.warmup_steps = warmup_steps
        self.active = None
        self.completed_steps = 0
        self.totals = defaultdict(lambda: [0, 0.0])

    def start_step(self, index):
        if self.active is not None:
            raise RuntimeError('a profiler update is already active')
        if type(index) is not int or index < 0:
            raise ValueError('update index must be a nonnegative integer')
        if index < self.warmup_steps:
            return
        tracing = len(self.traces) < self.trace_steps
        if not tracing and not (self.kernels and self.cuda):
            return
        activities = [torch.profiler.ProfilerActivity.CPU] if tracing else []
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities)
        profiler.__enter__()
        self.active = profiler
        self.active_trace, self.active_index = tracing, index

    def region(self, name):
        """Named CPU range only while this update is selected for Chrome export."""
        if self.active is not None and self.active_trace:
            return torch.profiler.record_function(name)
        return contextlib.nullcontext()

    def _export(self, profiler, index):
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        destination = self.trace_dir / f'step-{index:06d}.json'
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        descriptor, temporary = tempfile.mkstemp(prefix='.trace-', suffix='.json.tmp', dir=self.trace_dir)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            profiler.export_chrome_trace(str(temporary))
            digest = hashlib.sha256()
            with temporary.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                os.fsync(stream.fileno())
            # Hard-link publication is atomic and refuses an existing final path.
            os.link(temporary, destination)
            directory = os.open(self.trace_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return {'path': str(destination.resolve()), 'step_index': index,
                    'bytes': destination.stat().st_size, 'sha256': digest.hexdigest()}
        finally:
            temporary.unlink(missing_ok=True)

    def finish_step(self):
        if self.active is None:
            return
        profiler, self.active = self.active, None
        tracing, index = self.active_trace, self.active_index
        self.active_trace, self.active_index = False, None
        profiler.__exit__(None, None, None)
        # CPU operator aggregates include their children's device time. Counting
        # only raw CUDA events avoids counting that same kernel more than once.
        # Kineto can also emit CUDA user-annotation ranges spanning kernels;
        # exclude these even though their device_type is CUDA.
        pending_totals = defaultdict(lambda: [0, 0.0])
        for event in profiler.events() if self.kernels and self.cuda else ():
            if (event.device_type != torch.autograd.DeviceType.CUDA
                    or getattr(event, 'is_user_annotation', False)):
                continue
            duration = float(event.device_time_total)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError('invalid CUDA profiler event duration')
            entry = pending_totals[event.name]
            entry[0] += 1
            entry[1] += duration
        if tracing:
            self.traces.append(self._export(profiler, index))
        for name, (count, duration) in pending_totals.items():
            self.totals[name][0] += count
            self.totals[name][1] += duration
        self.completed_steps += 1

    def close(self):
        """Release an interrupted capture without reporting an incomplete update."""
        if self.active is not None:
            profiler, self.active = self.active, None
            self.active_trace, self.active_index = False, None
            profiler.__exit__(None, None, None)

    def report(self):
        total = sum(value[1] for value in self.totals.values())
        rows = [{'name': name, 'calls': count, 'cuda_time_us': duration,
                 'mean_cuda_us_per_call': duration / count,
                 'mean_cuda_ms_per_step': duration / self.completed_steps / 1000,
                 'fraction_of_device_time': duration / total if total else 0.0}
                for name, (count, duration) in self.totals.items()]
        rows.sort(key=lambda row: (-row['cuda_time_us'], row['name']))
        return {'traces': [dict(item) for item in self.traces], 'trace_count': len(self.traces),
                'trace_scope': 'first bounded complete post-warmup updates only; CPU activity plus optional CUDA; interrupted updates never exported',
                'completed_steps': self.completed_steps, 'total_cuda_time_us': total,
                'device_events_observed': sum(value[0] for value in self.totals.values()),
                'capture_status': ('kernel_ranking_disabled' if not self.kernels else
                                   'device_events_observed' if rows else 'no_device_events_observed'),
                'ranking': rows,
                'scope': 'raw CUDA device events (kernels and memory operations), summed over complete post-warmup updates; excludes checkpointing',
                'note': 'summed device duration is not elapsed wall time; concurrent events can overlap; profiler instrumentation perturbs throughput'}
