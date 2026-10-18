"""All-rank day-1 canary worker; launch with torchrun, not directly."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(result, rows, *, steps, world, peak_tflops_per_gpu, memory, wall_seconds):
    """Validate completion before reporting measured update and end-to-end rates."""
    if steps <= 0 or world < 2 or not math.isfinite(peak_tflops_per_gpu) or peak_tflops_per_gpu <= 0:
        raise ValueError('positive steps/peak TFLOPs and at least two ranks required')
    if len(rows) != steps or result['progress']['steps'] != steps:
        raise ValueError(f'canary incomplete: expected {steps} steps, observed {len(rows)}')
    for i, row in enumerate(rows, 1):
        if row['step'] != i:
            raise ValueError('canary metric steps are not contiguous')
        for field in ('loss', 'seconds', 'prediction_tokens', 'score_bound_max'):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f'nonfinite/negative {field} at step {i}')
        if row['seconds'] <= 0 or row['prediction_tokens'] <= 0:
            raise ValueError('canary must perform timed, nonempty updates')
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('invalid canary wall time')
    if len(memory) != world or sorted(m['rank'] for m in memory) != list(range(world)):
        raise ValueError('missing rank memory measurements')
    tokens = sum(r['prediction_tokens'] for r in rows)
    if tokens != result['progress']['tokens']:
        raise ValueError('metric token count does not match final progress')
    n_params = result['parameters']['total']
    if n_params <= 0:
        raise ValueError('invalid parameter count')
    def rate(selected):
        tok = sum(r['prediction_tokens'] for r in selected)
        seconds = sum(r['seconds'] for r in selected)
        aggregate = tok / seconds
        return {'steps': len(selected), 'prediction_tokens': tok, 'update_seconds': seconds,
                'tokens_per_second': aggregate, 'tokens_per_second_per_gpu': aggregate / world,
                'mfu_estimate': 6 * n_params * aggregate / (world * peak_tflops_per_gpu * 1e12)}
    # Exclude startup/first updates without claiming the LR warmup has ended.
    excluded = min(10, steps - 1)
    return {'status': 'passed', 'world_size': world, 'parameters_total': n_params,
            'all_updates': rate(rows), 'after_startup_updates': rate(rows[excluded:]),
            'startup_updates_excluded': excluded,
            'end_to_end_seconds': wall_seconds,
            'end_to_end_tokens_per_second': tokens / wall_seconds,
            'peak_tflops_per_gpu': peak_tflops_per_gpu,
            'mfu_definition': '6 * total parameters * global prediction tokens/s / (world_size * supplied peak FLOPs/s/GPU); approximate, excludes attention-specific FLOPs',
            'memory_per_gpu': memory,
            'loss': {'first': rows[0]['loss'], 'last': rows[-1]['loss'],
                     'min': min(r['loss'] for r in rows), 'max': max(r['loss'] for r in rows),
                     'sanity': 'all losses finite and nonnegative; no convergence claim'},
            'score_bound_max': max(r['score_bound_max'] for r in rows),
            'score_bound_scope': 'pre-update forward diagnostic, not a post-update score cap',
            'checkpoint': result['checkpoint']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--peak-tflops-per-gpu', type=float, required=True)
    p.add_argument('--run-budget-usd', type=float, required=True)
    a = p.parse_args(argv)
    import torch
    import torch.distributed as dist
    from tnlm.distributed_train import distributed_train_run
    from tnlm.distributed_checkpoint import _phase
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=300))
    rank = dist.get_rank()
    try:
        _phase('fresh canary output', lambda: a.output.mkdir(parents=True, exist_ok=False) if rank == 0 else None)
        started = time.monotonic()
        result = distributed_train_run(a.config, a.output / 'train', max_steps=a.steps,
                    max_seconds=a.max_seconds, device=f'cuda:{local_rank}', run_budget_usd=a.run_budget_usd)
        torch.cuda.synchronize()
        local_memory = {'rank': rank, 'local_rank': local_rank,
                        'device': torch.cuda.get_device_name(local_rank),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(local_rank),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(local_rank),
                        'total_bytes': torch.cuda.get_device_properties(local_rank).total_memory}
        memory = [None] * dist.get_world_size()
        dist.all_gather_object(memory, local_memory)
        elapsed = torch.tensor(time.monotonic() - started, device=f'cuda:{local_rank}', dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        def write_report():
            if rank != 0:
                return
            rows = [json.loads(line) for line in Path(result['metrics_path']).read_text().splitlines()]
            report = summarize(result, rows, steps=a.steps, world=dist.get_world_size(),
                peak_tflops_per_gpu=a.peak_tflops_per_gpu, memory=memory, wall_seconds=float(elapsed))
            (a.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        _phase('canary completion and measurements', write_report)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
