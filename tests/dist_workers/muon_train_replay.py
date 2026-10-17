"""Real two-rank Muon trainer: interrupted schedules and all state replay exactly."""
from datetime import timedelta
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import FixtureTokenizer
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data, RankPackedSampler
from tnlm.distributed_checkpoint import load_distributed_checkpoint
from tnlm.distributed_train import distributed_train_run
from tnlm.initialization import sharded_model_for_restore
from tnlm.train import make_optimizer


def equal(a, b):
    if isinstance(a, torch.Tensor):
        if hasattr(a, 'to_local'):
            a, b = a.to_local(), b.to_local()
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    if rank == 0:
        source = root / 'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefghij'}) + '\n' for i in range(30)))
        prepare_data(root / 'data', source_jsonl=source, revision='muon-trainer-v1',
            tokenizer=FixtureTokenizer(), split_fractions={'train': .8, 'calibration': .2, 'val': 0., 'test': 0.})
    dist.barrier()
    manifest = root / 'data' / 'manifest.json'
    mesh = init_device_mesh('cpu', (world,))
    for norm in ('pre_rms', 'spectral_rms', 'lagged'):
        cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
            d_mlp=12, normalization=norm, final_normalization='lagged' if norm == 'lagged' else 'rms',
            attention_backend='reference'),
            train=TrainConfig(optimizer='muon', compile=False, bf16_compute=False, lr=.01,
                aux_adamw_lr=.003, global_prediction_tokens_per_update=7, rung_total_tokens=100),
            raw={'data': {'manifest': str(manifest), 'context': 3, 'micro_batch_size': 1, 'calibration_tokens': 8},
                 'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0., 'spend_ledger': str(root / 'spend.json')}})
        def run(name, steps, **kwargs):
            return distributed_train_run(cfg, root / f'{norm}-{name}', max_steps=steps, max_seconds=120, **kwargs)
        def snapshot(result):
            model = sharded_model_for_restore(cfg.model, mesh, device=torch.device('cpu'))
            optimizer = make_optimizer(model, cfg.train)
            sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=3, seed=cfg.train.seed)
            load_distributed_checkpoint(result['checkpoint'], model, optimizer, sampler)
            state = get_state_dict(model, [optimizer.muon, optimizer.adamw])
            batch = sampler.next_batch()
            rng = torch.rand(8)
            return state, batch, rng
        whole = run('whole', 2)
        first = run('first', 1)
        resumed = run('resumed', 1, resume=first['checkpoint'])
        equal(whole['progress'], resumed['progress'])
        equal(snapshot(whole), snapshot(resumed))
        exposure = whole['progress']['optimizer_exposure']
        assert exposure['complete_history'] and exposure['observed_steps'] == 2
        for index, group in enumerate(exposure['groups']):
            assert group['sum_lr'] == sum(row['optimizer_group_lrs'][index] for row in whole['history'])
        assert len(whole['history']) == 2
        for step, row in enumerate(whole['history']):
            expected = [(step + 1) * .01 / 100, (step + 1) * .003 / 100, (step + 1) * .003 / 100]
            torch.testing.assert_close(torch.tensor(row['optimizer_group_lrs']), torch.tensor(expected), rtol=0, atol=1e-10)
        assert whole['history'][1]['rounds'] == resumed['history'][0]['rounds']
        cooled = run('cooled', 2, resume=first['checkpoint'], cooldown_steps=2)
        cool_first = run('cool-first', 1, resume=first['checkpoint'], cooldown_steps=2)
        cool_resume = run('cool-resume', 1, resume=cool_first['checkpoint'], cooldown_steps=2)
        equal(cooled['progress'], cool_resume['progress'])
        assert cooled['progress']['optimizer_exposure']['observed_steps'] == 3
        for index, group in enumerate(cooled['progress']['optimizer_exposure']['groups']):
            expected = first['progress']['optimizer_exposure']['groups'][index]['sum_lr']
            for row in cooled['history']:
                expected += row['optimizer_group_lrs'][index]
            assert group['sum_lr'] == expected
        equal(snapshot(cooled), snapshot(cool_resume))
        assert [r['optimizer_group_lrs'] for r in cooled['history']] == [r['optimizer_group_lrs'] for r in cool_first['history'] + cool_resume['history']]
        assert cooled['progress']['distributed_schedule']['branch_base_lr'] == first['history'][0]['optimizer_group_lrs']
        if rank == 0:
            print(f'{norm}: Muon training and cooldown exact state/data/RNG replay passed', flush=True)
    cfg.train.aux_adamw_lr = 0.
    try:
        run('zero-auxiliary-lr', 1)
    except ValueError as error:
        assert 'zero auxiliary AdamW LR' in str(error)
    else:
        raise AssertionError('silently substituted zero auxiliary LR')
    assert not (root / f'{norm}-zero-auxiliary-lr').exists()
    if rank == 0:
        ledger = json.loads((root / 'spend.json').read_text())
        assert len(ledger['runs']) == 18
        assert all(row['status'] == 'complete' for row in ledger['runs'].values())
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
