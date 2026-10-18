"""torchrun smoke: tiny FSDP training versus an interrupted, exact resumed run.

NCCL is the default; pass --backend gloo for an explicit CPU-only test. Outputs
are created exclusively under a new --output directory. CUDA kernel equivalence
and large-model throughput are separate day-1 checks, not claims of this smoke.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compare_tree(left, right):
    """Compare checkpoint trees exactly, also reporting finite numeric max error."""
    import numpy as np
    import torch
    result = {'exact': True, 'max_abs_diff': 0.0, 'mismatch_count': 0, 'first_mismatches': []}
    def mismatch(path):
        result['exact'] = False
        result['mismatch_count'] += 1
        if len(result['first_mismatches']) < 16:
            result['first_mismatches'].append(path)
    def visit(a, b, path):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                mismatch(path)
                return
            if not torch.equal(a, b):
                mismatch(path)
                if a.numel():
                    value = float((a.detach().double().cpu() - b.detach().double().cpu()).abs().max())
                    if result['max_abs_diff'] is not None:
                        result['max_abs_diff'] = max(result['max_abs_diff'], value) if math.isfinite(value) else None
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                mismatch(path + '.keys')
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                mismatch(path + '.shape')
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{path}[{i}]')
        elif type(a) is not type(b) or a != b:
            mismatch(path)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                value = abs(a-b)
                if result['max_abs_diff'] is not None:
                    result['max_abs_diff'] = max(result['max_abs_diff'], value) if math.isfinite(value) else None
    visit(left, right, 'root')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    parser.add_argument('--hourly-rate', type=float)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.steps < 2 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('steps must be >=2 and max-seconds must be finite and positive')
    rate = args.hourly_rate if args.hourly_rate is not None else (0.0 if args.backend == 'gloo' else None)
    if rate is None or not math.isfinite(rate) or rate < 0 or (args.backend == 'nccl' and rate <= 0):
        parser.error('NCCL requires --hourly-rate with the actual positive whole-job hourly rate')
    # Must precede CUDA context creation by NCCL or device-mesh setup.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    from tnlm.config import Config, ModelConfig, TrainConfig
    from tnlm.data import RankPackedSampler, prepare_data
    from tnlm.distributed_checkpoint import _phase, load_distributed_checkpoint
    from tnlm.distributed_train import distributed_train_run
    from tnlm.initialization import sharded_model_for_restore
    from tnlm.parallel import full_state_dict
    from tnlm.train import make_optimizer
    from tnlm.runtime import deterministic_execution

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    device = torch.device('cpu')
    if args.backend == 'nccl':
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            parser.error('NCCL smoke requires CUDA and an NCCL-enabled PyTorch build')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=max(120, args.max_seconds)))
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(args.output).resolve()
    created = False
    report = {'status': 'failed', 'backend': args.backend, 'world_size': world,
              'steps': args.steps, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'scope': 'same-world-size tiny eager FSDP replay; synthetic GPT2 data; not kernel equivalence or a throughput canary'}
    try:
        _phase('fresh smoke output', lambda: output.mkdir(parents=True, exist_ok=False) if rank == 0 else None)
        created = True
        def make_data():
            source = output/'source.jsonl'
            source.write_text(''.join(json.dumps({'id': str(i), 'text':
                ('Deterministic distributed smoke document %d. ' % i) * (1+i % 7)})+'\n' for i in range(100)))
            prepare_data(output/'data', source_jsonl=source, revision='synthetic-h100-smoke-v1',
                         max_tokens=50000, max_documents=100,
                         split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
        _phase('real GPT2 smoke data', make_data if rank == 0 else lambda: None)
        manifest = output/'data/manifest.json'
        batch = world * 16 + 1
        cfg = Config(model=ModelConfig(layers=1, d_model=16, heads=2, d_key=8, d_value=8, d_mlp=32,
                       padded_vocab=50304, attention_backend='reference'),
                     train=TrainConfig(compile=False, deterministic=True, bf16_compute=device.type == 'cuda',
                       lr=.001, global_prediction_tokens_per_update=batch, rung_total_tokens=args.steps * batch - 1,
                       qk_clip_source='batch_bound'),
                     raw={'data': {'manifest': str(manifest), 'context': 16, 'micro_batch_size': 1},
                          'project': {'dry_run': False, 'gpu_hourly_rate_usd': rate,
                           'spend_ledger': str(output/'spend.json'), 'reserve_usd': 0,
                           'total_budget_usd': max(1., rate * args.max_seconds / 3600 * 4),
                           'keep_recovery_checkpoints': 0, 'save_initial_checkpoint': False,
                           'milestone_tokens': []}})
        half = args.steps // 2
        whole = distributed_train_run(cfg, output/'whole', max_steps=args.steps, max_seconds=args.max_seconds, device=str(device))
        first = distributed_train_run(cfg, output/'first', max_steps=half, max_seconds=args.max_seconds, device=str(device))
        resumed = distributed_train_run(cfg, output/'resumed', max_steps=args.steps-half, max_seconds=args.max_seconds,
                                        device=str(device), resume=first['checkpoint'])
        if (whole['progress']['steps'] != args.steps or first['progress']['steps'] != half
                or resumed['progress']['steps'] != args.steps):
            raise RuntimeError('smoke stopped early; increase --max-seconds before interpreting replay')
        mesh = init_device_mesh(device.type, (world,))
        def snapshot(result):
            model = sharded_model_for_restore(cfg.model, mesh, device)
            optimizer = make_optimizer(model, cfg.train)
            sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=16, seed=cfg.train.seed)
            with deterministic_execution(True, device=str(device)):
                payload = load_distributed_checkpoint(result['checkpoint'], model, optimizer, sampler)
            state = {'model': full_state_dict(model), 'optimizer': get_optimizer_state_dict(model, optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True)),
                'sampler': sampler.state_dict(), 'progress': payload['progress'], 'rng': payload['rng']}
            return state
        left = snapshot(whole)
        gc.collect()
        right = snapshot(resumed)
        comparisons = {name: compare_tree(left[name], right[name]) for name in left}
        fields = ('step', 'tokens', 'prediction_tokens', 'loss', 'grad_norm', 'lr', 'rounds', 'score_bound_max')
        comparisons['metrics'] = compare_tree([{k: r[k] for k in fields} for r in whole['history']],
            [{k: r[k] for k in fields} for r in first['history'] + resumed['history']])
        gathered = [None] * world
        dist.all_gather_object(gathered, comparisons)
        report['comparisons_by_rank'] = gathered
        report['exact_resume'] = all(item['exact'] for comparison in gathered for item in comparison.values())
        report['max_abs_diffs'] = {name: (None if any(c[name]['max_abs_diff'] is None for c in gathered)
            else max(c[name]['max_abs_diff'] for c in gathered)) for name in comparisons}
        report['checkpoints'] = {name: value['checkpoint'] for name, value in
                                 (('whole', whole), ('first', first), ('resumed', resumed))}
        report['progress'] = whole['progress']
        report['device'] = str(device)
        report['deterministic'] = True
        report['status'] = 'passed' if report['exact_resume'] else 'failed'
        if not report['exact_resume']:
            report['error'] = 'interrupted/resumed state or trajectory differs from uninterrupted training'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if rank == 0:
            if created:
                (output/'result.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
            print(json.dumps({'status': report['status'], 'report': str(output/'result.json') if created else None,
                              'error': report.get('error')}, sort_keys=True), flush=True)
        dist.destroy_process_group()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
