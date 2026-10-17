"""A bounded throughput gate over real, completed training updates."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile


class ThroughputCanary:
    """Measure N global optimizer steps after a declared warmup boundary.

    A passed gate says nothing about subsequent completion or model quality.
    Observation never changes training tensors, optimizers, samplers or RNG.
    """

    def __init__(self, measured_steps, min_tokens_per_second, warmup_steps, *, output_path=None):
        if type(measured_steps) is not int or measured_steps <= 0:
            raise ValueError('measured_steps must be a positive integer')
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError('warmup_steps must be a nonnegative integer')
        if (type(min_tokens_per_second) not in (int, float) or
                not math.isfinite(min_tokens_per_second) or min_tokens_per_second <= 0):
            raise ValueError('min_tokens_per_second must be finite and positive')
        self.measured_steps = measured_steps
        self.min_tokens_per_second = float(min_tokens_per_second)
        self.warmup_steps = warmup_steps
        self.output_path = None
        self.records = []
        self.status = 'pending'
        self.last_step = 0
        if output_path is not None:
            self.bind_output(output_path)

    @property
    def required_updates(self):
        return self.warmup_steps + self.measured_steps

    def bind_output(self, path):
        path = Path(path)
        if self.output_path is not None:
            if path.absolute() != self.output_path.absolute():
                raise ValueError('canary output path cannot change')
            return
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        self.output_path = path

    def report(self):
        tokens = sum(row['prediction_tokens'] for row in self.records)
        seconds = sum(row['seconds'] for row in self.records)
        return {
            'kind': 'throughput_canary', 'status': self.status,
            'controls': {'measured_steps': self.measured_steps,
                         'warmup_steps': self.warmup_steps,
                         'min_tokens_per_second': self.min_tokens_per_second},
            'selected_steps': list(self.records), 'completed_measured_steps': len(self.records),
            'prediction_tokens': tokens, 'training_seconds': seconds,
            'tokens_per_second': tokens / seconds if seconds else None,
            'timing_scope': 'ratio of summed actual prediction targets to summed synchronized completed-update seconds; includes accumulation, forward/backward and optimizer; excludes checkpoint writes',
            'scope': 'throughput gate only; passed does not imply full-run completion, future throughput or model quality',
        }

    def _publish(self):
        if self.output_path is None:
            return
        path = self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', prefix='.' + path.name + '.',
                    dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.report(), stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def observe(self, record):
        if self.status == 'passed':
            return
        if self.status in ('below_threshold', 'insufficient'):
            raise RuntimeError('throughput canary already failed')
        step = record.get('step')
        if type(step) is not int or step <= self.last_step:
            raise ValueError('canary requires strictly increasing positive global steps')
        self.last_step = step
        if step <= self.warmup_steps:
            return
        tokens, seconds = record.get('prediction_tokens'), record.get('seconds')
        if (type(tokens) is not int or tokens <= 0 or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('canary requires positive targets and finite positive update seconds')
        self.records.append({'step': step, 'prediction_tokens': tokens, 'seconds': float(seconds)})
        self.status = 'collecting'
        if len(self.records) == self.measured_steps:
            self.status = ('passed' if self.report()['tokens_per_second'] >= self.min_tokens_per_second
                           else 'below_threshold')
        self._publish()
        if self.status == 'below_threshold':
            raise RuntimeError('throughput canary below minimum tokens per second')

    def require_complete(self):
        if self.status != 'passed':
            self.status = 'insufficient'
            self._publish()
            raise RuntimeError('throughput canary has insufficient completed measured steps')
        return self.report()
