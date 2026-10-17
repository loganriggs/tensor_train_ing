"""Two-rank Muon acceptance; FP32 parameters, native BF16 Newton-Schulz."""
import json
from pathlib import Path
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import FixtureTokenizer
from tnlm.config import ModelConfig
from tnlm.data import RankPackedSampler, batch_hash, manifest_hash, prepare_data
from tnlm.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules, spectral_step
from tnlm.optim import MuonAdamW
from tnlm.parallel import global_count, shard_model


# Declared before test execution. Prior independent three-step tiny-matrix probes
# observed 1.3e-4..2.6e-4 absolute differences due to BF16 NS reduction order.
# This is numerical global-matrix equivalence, not bitwise serial-seed identity.
SERIAL_WEIGHT_ATOL = 4e-4


def clone(value):
    if isinstance(value, torch.Tensor):
        return (value.to_local() if isinstance(value, DTensor) else value).detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return value


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), (left.shape, right.shape)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def logical_matrix_checks(mesh, rank):
    measurements = []
    for shape in ((8, 4), (4, 8), (7, 5)):
        generator = torch.Generator().manual_seed(137)
        initial = torch.randn(shape, generator=generator)
        sharded = torch.nn.Parameter(distribute_tensor(initial.clone(), mesh, [Shard(0)]))
        serial = torch.nn.Parameter(initial.clone())
        naive_local = torch.nn.Parameter(sharded.to_local().clone())
        optimizers = [torch.optim.Muon([parameter], lr=.01, adjust_lr_fn='match_rms_adamw')
                      for parameter in (sharded, serial, naive_local)]
        for step in range(3):
            gradient = torch.randn(shape, generator=generator)
            sharded.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
            serial.grad = gradient.clone()
            naive_local.grad = gradient.chunk(2, dim=0)[rank].clone()
            for optimizer in optimizers:
                optimizer.step()
        actual = sharded.full_tensor()
        torch.testing.assert_close(actual, serial, atol=SERIAL_WEIGHT_ATOL, rtol=0)
        momentum = optimizers[0].state[sharded]['momentum_buffer'].full_tensor()
        torch.testing.assert_close(momentum, optimizers[1].state[serial]['momentum_buffer'], rtol=0, atol=0)
        shards = [None] * 2
        dist.all_gather_object(shards, naive_local.detach())
        naive = torch.cat(shards)
        measured = {'shape': list(shape),
                    'dtensor_vs_serial_max_abs': float((actual - serial).detach().abs().max()),
                    'naive_local_vs_serial_max_abs': float((naive - serial).detach().abs().max()),
                    'naive_local_vs_dtensor_max_abs': float((naive - actual).detach().abs().max()),
                    'momentum_vs_serial_max_abs': float((momentum - optimizers[1].state[serial]['momentum_buffer']).detach().abs().max())}
        assert measured['naive_local_vs_serial_max_abs'] > SERIAL_WEIGHT_ATOL
        assert measured['naive_local_vs_dtensor_max_abs'] > SERIAL_WEIGHT_ATOL
        measurements.append(measured)
        if rank == 0:
            print(f'{shape[0]}x{shape[1]}: logical global-matrix Muon within fixed BF16 tolerance', flush=True)
            print('Muon numerical evidence: ' + json.dumps(measured, sort_keys=True), flush=True)
    return measurements


def update(model, optimizer, batch, rank, *, zero_rank=False):
    optimizer.zero_grad(set_to_none=True)
    spectral_step(model)
    idx, docs, targets = batch
    targets = targets.clone()
    if rank == 0:
        if zero_rank:
            targets.fill_(-100)
        else:
            positions = (targets.flatten() != -100).nonzero().flatten()
            targets.flatten()[positions[1:]] = -100
    for norm in lagged_modules(model):
        norm.collecting = True
        norm.set_valid_mask(docs >= 0)
    try:
        _, loss, count = model(idx, docs, targets, reduction='sum')
        total = global_count(count)
        assert total > 0
        (loss / total).backward()
    finally:
        for norm in lagged_modules(model):
            norm.collecting = False
            norm.set_valid_mask(None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    for norm in lagged_modules(model):
        norm.commit()
    optimizer.zero_grad(set_to_none=True)


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh('cpu', (world,))
    measurements = logical_matrix_checks(mesh, rank)
    root = Path(sys.argv[1])
    if rank == 0:
        (root / 'muon-semantics.json').write_text(json.dumps({'torch': str(torch.__version__),
            'serial_weight_atol': SERIAL_WEIGHT_ATOL, 'serial_weight_rtol': 0, 'measurements': measurements}, indent=2) + '\n')
    if rank == 0:
        source = root / 'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefghijklm'}) + '\n' for i in range(12)))
        prepare_data(root / 'data', source_jsonl=source, revision='muon-replay-fixture', tokenizer=FixtureTokenizer(),
                     split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    dist.barrier()
    manifest = root / 'data/manifest.json'
    for package in ('pre_rms', 'spectral_rms', 'lagged'):
        torch.manual_seed(81)
        cfg = ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                          normalization=package, final_normalization='lagged' if package == 'lagged' else 'rms',
                          padded_vocab=50257, attention_backend='reference', loss_chunk_tokens=2)
        model = shard_model(TensorLM(cfg), mesh=mesh)
        for unit in [model, *model.blocks]:
            unit.set_force_sum_reduction_for_comms(True)
        optimizer = MuonAdamW(model, lr=.001, aux_lr=.001, weight_decay=.1, betas=(.9, .95))
        sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=5, seed=17)
        settings = {'optimizer': 'muon', 'normalization': package}
        initial_path = root / f'{package}-initial'
        original_model = clone(model.state_dict())
        original_rng = torch.get_rng_state().clone()
        # Saving must not enter either child step, including for empty optimizer state.
        original_steps = optimizer.muon.step, optimizer.adamw.step
        def forbidden_step(*args, **kwargs):
            raise AssertionError('checkpoint invoked optimizer.step')
        optimizer.muon.step = optimizer.adamw.step = forbidden_step
        try:
            save_distributed_checkpoint(initial_path, model, optimizer, sampler, config=settings,
                                       progress={'steps': 0}, manifest_hash=manifest_hash(manifest))
        finally:
            optimizer.muon.step, optimizer.adamw.step = original_steps
        equal(original_model, clone(model.state_dict()))
        assert torch.equal(torch.get_rng_state(), original_rng)
        assert optimizer.muon.state and optimizer.adamw.state
        for state in optimizer.muon.state.values():
            assert set(state) == {'momentum_buffer'}
            assert torch.count_nonzero(state['momentum_buffer'].to_local()) == 0
        assert all(float(state['step']) == 0 for state in optimizer.adamw.state.values())
        l, r = model.blocks[0].mlp.l.weight, model.blocks[0].mlp.r.weight
        assert l is not r
        assert sum(parameter is l for group in optimizer.muon.param_groups for parameter in group['params']) == 1
        assert sum(parameter is r for group in optimizer.muon.param_groups for parameter in group['params']) == 1
        assert optimizer.muon.state[l]['momentum_buffer'] is not optimizer.muon.state[r]['momentum_buffer']
        # Initial checkpoint restoration must preserve first genuine AdamW step.
        load_distributed_checkpoint(initial_path, model, optimizer, sampler, expected_config=settings)
        update(model, optimizer, sampler.next_batch(), rank)
        assert all(float(state['step']) == 1 for state in optimizer.adamw.state.values())
        checkpoint = root / f'{package}-step1'
        save_distributed_checkpoint(checkpoint, model, optimizer, sampler, config=settings,
                                   progress={'steps': 1}, manifest_hash=manifest_hash(manifest))
        expected_batch = sampler.next_batch()
        update(model, optimizer, expected_batch, rank, zero_rank=True)
        expected = clone({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        sampler.next_batch()
        payload = load_distributed_checkpoint(checkpoint, model, optimizer, sampler, expected_config=settings,
                                             expected_manifest_hash=manifest_hash(manifest))
        assert payload['progress'] == {'steps': 1}
        replay = sampler.next_batch()
        assert batch_hash(replay) == batch_hash(expected_batch)
        update(model, optimizer, replay, rank, zero_rank=True)
        equal(expected, clone({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}))
        before = clone(model.state_dict())
        changed = torch.optim.AdamW(model.parameters(), lr=.001)
        try:
            load_distributed_checkpoint(checkpoint, model, changed, sampler)
        except ValueError as error:
            assert 'layout' in str(error) or 'optimizer' in str(error)
        else:
            raise AssertionError('changed optimizer kind accepted')
        equal(before, clone(model.state_dict()))
        optimizer.muon.param_groups[0]['params'].reverse()
        try:
            load_distributed_checkpoint(checkpoint, model, optimizer, sampler)
        except ValueError as error:
            assert 'layout' in str(error) or 'optimizer' in str(error)
        else:
            raise AssertionError('changed hidden matrix group order accepted')
        finally:
            optimizer.muon.param_groups[0]['params'].reverse()
        equal(before, clone(model.state_dict()))
        if rank == 0:
            print(f'{package}: initial state and exact Muon/AdamW checkpoint replay', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
