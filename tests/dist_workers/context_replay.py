"""Actual two-rank context-fork replay, including a pending transition checkpoint."""
import copy
import hashlib
from datetime import timedelta
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import FixtureTokenizer
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data, RankPackedSampler, batch_hash
from tnlm.distributed_checkpoint import load_distributed_checkpoint
from tnlm.distributed_train import distributed_train_run
from tnlm.parallel import shard_model, full_state_dict
from tnlm.model import TensorLM
from tnlm.train import make_optimizer
from tnlm.evaluate import _training_descriptor


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b, (a, b)


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=150))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    mesh = init_device_mesh('cpu', (world,))
    if rank == 0:
        source = root / 'source.jsonl'
        ids = {}
        for i in range(100):
            name = f'doc-{i}'
            split = 'train' if int(hashlib.sha256(name.encode()).hexdigest(), 16) < int(.8 * 2**256) else 'calibration'
            ids.setdefault(split, name)
        source.write_text(''.join(json.dumps({'id': ids[split], 'text': text * 300}) + '\n'
            for split, text in [('train', 'abcdefgh'), ('calibration', 'ijklmnop')]))
        prepare_data(root / 'data', source_jsonl=source, revision='context-replay-v1',
                     max_tokens=4000, tokenizer=FixtureTokenizer(),
                     split_fractions={'train': .8, 'val': 0., 'test': 0., 'calibration': .2})
    dist.barrier()
    manifest = root / 'data' / 'manifest.json'
    for optimizer_kind, normalization in [('adamw', 'lagged'), ('muon', 'spectral_rms')]:
        case = root / optimizer_kind
        cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
            vocab_size=128, padded_vocab=128, attention_backend='reference', normalization=normalization,
            final_normalization='lagged' if normalization == 'lagged' else 'rms'),
            train=TrainConfig(optimizer=optimizer_kind, compile=False, bf16_compute=False,
                             lr=.01, aux_adamw_lr=.003, rung_total_tokens=101,
                             global_prediction_tokens_per_update=7),
            raw={'data': {'manifest': str(manifest), 'context': 2, 'micro_batch_size': 1, 'calibration_tokens': 8},
                 'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0., 'spend_ledger': str(case / 'ledger.json'),
                             'milestone_tokens': [], 'checkpoint_every_seconds': 1000000}})
        old = distributed_train_run(cfg, case / 'old', max_steps=1, max_seconds=120)
        assert old['progress']['context_history'] == [{'context': 2, 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}]
        new = copy.deepcopy(cfg)
        new.raw['data']['context'] = 7
        whole = distributed_train_run(new, case / 'whole', max_steps=3, max_seconds=120,
                                     resume=old['checkpoint'], context_fork=True)
        first = distributed_train_run(new, case / 'first', max_steps=1, max_seconds=120,
                                     resume=old['checkpoint'], context_fork=True)
        resumed = distributed_train_run(new, case / 'resumed', max_steps=2, max_seconds=120,
                                       resume=first['checkpoint'])
        pending = distributed_train_run(new, case / 'pending', max_steps=3, max_seconds=120,
                                       resume=case / 'whole' / 'initial')
        exact(whole['progress'], resumed['progress'])
        exact(whole['progress'], pending['progress'])
        assert whole['progress']['tokens'] == 28
        assert whole['history'][0]['rounds'][0]['raw_counts'] == [6, 7]
        assert whole['history'][0]['rounds'][0]['selected_counts'] == [6, 1]
        exposure = whole['progress']['context_fork_exposure']
        assert exposure['prefix_tokens_at_or_before_watermark'] == 2
        assert exposure['masked_prediction_targets'] == 1
        loaded = []
        for result in [whole, resumed, pending]:
            model = shard_model(TensorLM(new.model), mesh)
            optimizer = make_optimizer(model, new.train)
            sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=7, seed=new.train.seed)
            payload = load_distributed_checkpoint(result['checkpoint'], model, optimizer, sampler)
            descriptor = _training_descriptor({'config': payload['config'], 'progress': payload['progress']})
            assert result['training_descriptor'] == descriptor
            assert descriptor['context_history_status'] == 'complete' and descriptor['train_context'] == 7
            assert descriptor['completion_status'] == 'partial'
            children = [optimizer.muon, optimizer.adamw] if optimizer_kind == 'muon' else optimizer
            loaded.append((full_state_dict(model), get_optimizer_state_dict(model, children,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True)), sampler.state_dict(),
                batch_hash(sampler.next_batch()), torch.rand(4)))
        exact(loaded[0], loaded[1])
        exact(loaded[0], loaded[2])
        source_model = shard_model(TensorLM(cfg.model), mesh)
        source_optimizer = make_optimizer(source_model, cfg.train)
        source_sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=2, seed=cfg.train.seed)
        source_payload = load_distributed_checkpoint(old['checkpoint'], source_model, source_optimizer, source_sampler)
        source_children = [source_optimizer.muon, source_optimizer.adamw] if optimizer_kind == 'muon' else source_optimizer
        source_snapshot = (full_state_dict(source_model), get_optimizer_state_dict(source_model, source_children,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True)), torch.rand(4))
        initial_model = shard_model(TensorLM(new.model), mesh)
        initial_optimizer = make_optimizer(initial_model, new.train)
        initial_sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=7, seed=new.train.seed)
        initial_payload = load_distributed_checkpoint(case / 'whole' / 'initial', initial_model, initial_optimizer, initial_sampler)
        initial_children = [initial_optimizer.muon, initial_optimizer.adamw] if optimizer_kind == 'muon' else initial_optimizer
        initial_snapshot = (full_state_dict(initial_model), get_optimizer_state_dict(initial_model, initial_children,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True)), torch.rand(4))
        exact(source_snapshot, initial_snapshot)
        for key in ('steps', 'tokens', 'scheduler_step', 'warmup_steps', 'optimizer_exposure', 'calibration'):
            assert source_payload['progress'].get(key) == initial_payload['progress'].get(key)
        entry = initial_payload['progress']['context_history'][-1]
        assert entry['start_tokens'] == 7 and entry['start_step'] == 1
        assert entry['source_sampler'] == source_sampler.state_dict()['stream']
        assert 'context_transition' in initial_sampler.state_dict()['stream']
        before = (case / 'ledger.json').read_bytes()
        invalid = [('missing-parent', new, {'context_fork': True}),
                   ('nonbool', new, {'context_fork': 1, 'resume': old['checkpoint']}),
                   ('no-opt-in', new, {'resume': old['checkpoint']}),
                   ('equal', cfg, {'context_fork': True, 'resume': old['checkpoint']})]
        smaller = copy.deepcopy(cfg); smaller.raw['data']['context'] = 1
        invalid.append(('smaller', smaller, {'context_fork': True, 'resume': old['checkpoint']}))
        changed = copy.deepcopy(new); changed.train.global_prediction_tokens_per_update = 8
        invalid.append(('batch', changed, {'context_fork': True, 'resume': old['checkpoint']}))
        for name, bad, kwargs in invalid:
            try:
                distributed_train_run(bad, case / name, max_steps=1, max_seconds=120, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(name + ' accepted')
            assert not (case / name).exists()
        assert (case / 'ledger.json').read_bytes() == before
        if rank == 0:
            print(optimizer_kind + ': exact context-fork pending/replay/norm/RNG and prebilling rejection passed', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
