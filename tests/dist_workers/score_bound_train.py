"""Two-rank recording-only score bounds do not change the training trajectory."""
from collections import defaultdict
from datetime import timedelta
import importlib
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import RankPackedSampler, prepare_data
from tnlm.distributed_checkpoint import load_distributed_checkpoint
from tnlm.model import TensorLM
from tnlm.parallel import full_state_dict, shard_model
from tnlm.train import make_optimizer
from train_replay import flatten_equal


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    class Tokenizer:
        name, eot_token = 'distributed-score-fixture', 50256
        def encode(self, text, **kwargs):
            return list(text.encode())
    if rank == 0:
        source = root/'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': ['ab', 'abcde', 'abcdefg'][i % 3]})+'\n' for i in range(30)))
        prepare_data(root/'data', source_jsonl=source, revision='score-test', tokenizer=Tokenizer(),
                     split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    dist.barrier()
    manifest = root/'data/manifest.json'
    cfg = Config(model=ModelConfig(layers=1, d_model=4, heads=1, d_key=2, d_value=2, d_mlp=8,
                     padded_vocab=50257, attention_backend='reference'),
                 train=TrainConfig(compile=False, bf16_compute=False, lr=.001,
                     global_prediction_tokens_per_update=7, rung_total_tokens=8),
                 raw={'data': {'manifest': str(manifest), 'context': 3, 'micro_batch_size': 1},
                      'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0., 'spend_ledger': str(root/'spend.json'),
                                  'keep_recovery_checkpoints': 0, 'save_initial_checkpoint': False,
                                  'milestone_tokens': []}})
    trainer = importlib.import_module('tnlm.distributed_train')
    model_module = importlib.import_module('tnlm.model')
    original_bound, original_phase = model_module.score_bound, trainer._phase
    step = 0
    seen = defaultdict(list)
    def observed_bound(q, k, dk):
        bound = original_bound(q, k, dk) * (rank + 1)
        if step == 0 and not seen[step]:
            bound = bound * 1000  # Early-round maximum must survive later rounds.
        seen[step].append(float(bound.max()))
        return bound
    def observed_phase(name, function):
        nonlocal step
        result = original_phase(name, function)
        if name == 'optimizer update':
            step += 1
        return result
    model_module.score_bound = observed_bound
    trainer._phase = observed_phase
    cfg.train.qk_clip_source = 'batch_bound'
    recorded = trainer.distributed_train_run(cfg, root/'recorded', max_steps=2, max_seconds=120)
    assert len(recorded['history'][0]['rounds']) > 1
    assert any(0 in rd['selected_counts'] for row in recorded['history'] for rd in row['rounds'])
    for index, row in enumerate(recorded['history']):
        expected = torch.tensor(max(seen[index]), dtype=torch.float64)
        dist.all_reduce(expected, op=dist.ReduceOp.MAX)
        assert row['score_bound_max'] == float(expected)
    model_module.score_bound, trainer._phase = original_bound, original_phase
    cfg.train.qk_clip_source = 'probe'
    plain = trainer.distributed_train_run(cfg, root/'plain', max_steps=2, max_seconds=120)
    assert all('score_bound_max' not in row for row in plain['history'])
    mesh = init_device_mesh('cpu', (world,))
    def state(result):
        model = shard_model(TensorLM(cfg.model), mesh)
        opt = make_optimizer(model, cfg.train)
        sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=3, seed=cfg.train.seed)
        payload = load_distributed_checkpoint(result['checkpoint'], model, opt, sampler)
        return (full_state_dict(model), get_optimizer_state_dict(model, opt,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True)), sampler.state_dict(), payload['progress'])
    a, b = state(recorded), state(plain)
    if rank == 0:
        flatten_equal(a, b)
    cfg.train.qk_clip_source = 'batch_bound'
    def nonfinite_bound(q, k, dk):
        value = original_bound(q, k, dk)
        return value * float('inf') if rank == 1 else value
    model_module.score_bound = nonfinite_bound
    try:
        trainer.distributed_train_run(cfg, root/'nonfinite', max_steps=1, max_seconds=120)
    except (ValueError, FloatingPointError, RuntimeError) as error:
        assert 'finite' in str(error).lower(), str(error)
    else:
        raise AssertionError('non-finite rank-local score bound accepted')
    finally:
        model_module.score_bound = original_bound
    cfg.train.qk_clip_tau, cfg.train.qk_clip_every_steps = 100., 1
    try:
        trainer.distributed_train_run(cfg, root/'active', max_steps=1, max_seconds=120)
    except ValueError as error:
        assert 'serial' in str(error)
    else:
        raise AssertionError('active distributed QK clip accepted')
    assert not (root/'active').exists()
    if rank == 0:
        print('Distributed score bounds: global maxima, unchanged trajectory, collective failure passed', flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
