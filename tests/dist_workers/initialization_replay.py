"""Tiny two-rank CPU test: bounded restore allocation versus ordinary destination."""
import json
from pathlib import Path
import sys
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tnlm.initialization as initialization
from tnlm.config import ModelConfig
from tnlm.data import RankPackedSampler, manifest_hash, prepare_data
from tnlm.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules, spectral_step
from tnlm.parallel import global_count, shard_model


class Tokenizer:
    name, eot_token = 'offline', 50256
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def identical(left, right):
    if isinstance(left, torch.Tensor):
        a = left.to_local() if isinstance(left, DTensor) else left
        b = right.to_local() if isinstance(right, DTensor) else right
        assert torch.equal(a, b), (a.shape, b.shape)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            identical(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            identical(a, b)
    else:
        assert left == right


def update(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    spectral_step(model)
    idx, docs, targets = batch
    for norm in lagged_modules(model):
        norm.collecting = True
        norm.set_valid_mask(docs >= 0)
    try:
        _, loss, count = model(idx, docs, targets, reduction='sum')
        (loss / global_count(count)).backward()
    finally:
        for norm in lagged_modules(model):
            norm.collecting = False
            norm.set_valid_mask(None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), .25, error_if_nonfinite=True)
    optimizer.step()
    for norm in lagged_modules(model):
        norm.commit()
    optimizer.zero_grad(set_to_none=True)


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    if rank == 0:
        source = root / 'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefghi'}) + '\n' for i in range(8)))
        prepare_data(root / 'data', source_jsonl=source, revision='restore-fixture-v1', tokenizer=Tokenizer(),
                     split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    dist.barrier()
    manifest = root / 'data/manifest.json'
    mesh = init_device_mesh('cpu', (world,))
    for case in ('pre_rms', 'spectral_rms', 'lagged', 'tied'):
        package = 'pre_rms' if case == 'tied' else case
        cfg = ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                          padded_vocab=50257, normalization=package,
                          final_normalization='lagged' if package == 'lagged' else 'rms',
                          tie_embeddings=case == 'tied', attention_backend='reference')
        torch.manual_seed(87)
        normal = shard_model(TensorLM(cfg), mesh=mesh)
        for unit in [normal, *normal.blocks]:
            unit.set_force_sum_reduction_for_comms(True)
        optimizer = torch.optim.AdamW(normal.parameters(), lr=.001)
        sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=4, seed=11)
        update(normal, optimizer, sampler.next_batch())
        settings = {'case': case}
        checkpoint = root / case
        save_distributed_checkpoint(checkpoint, normal, optimizer, sampler,
                                   config=settings, progress={'steps': 1}, manifest_hash=manifest_hash(manifest))
        constructed = []
        original_reset = TensorLM.reset_parameters
        original_parameter_new = torch.nn.Parameter.__new__
        original_shard = initialization.shard_model

        def checked_reset(model):
            assert all(t.is_meta for t in (*model.parameters(), *model.buffers()))
            constructed.append(True)
            return original_reset(model)

        def checked_parameter_new(cls, data=None, requires_grad=True):
            assert data is None or data.is_meta, 'constructor allocated real unsharded parameter storage'
            return original_parameter_new(cls, data, requires_grad)

        def checked_shard(model, **kwargs):
            assert all(t.is_meta for t in (*model.parameters(), *model.buffers()))
            # Restrict the Parameter-constructor spy to TensorLM construction;
            # FSDP may legitimately create real local DTensor storage afterward.
            return original_shard(model, **kwargs)

        original_constructor = initialization.TensorLM
        def checked_constructor(configuration):
            with patch.object(torch.nn.Parameter, '__new__', staticmethod(checked_parameter_new)):
                return original_constructor(configuration)

        with patch.object(initialization, 'TensorLM', checked_constructor), \
             patch.object(TensorLM, 'reset_parameters', checked_reset), \
             patch.object(initialization, 'shard_model', checked_shard):
            restored = initialization.sharded_model_for_restore(cfg, mesh, 'cpu')
        assert constructed == [True]  # no reset_parameters after materialization
        for parameter in restored.parameters():
            assert isinstance(parameter, DTensor)
            assert torch.count_nonzero(parameter.to_local()) == 0
            assert parameter.to_local().numel() <= (parameter.numel() + world - 1) // world + cfg.d_model
        for buffer in restored.buffers():
            assert torch.count_nonzero(buffer) == 0
        if cfg.tie_embeddings:
            assert restored.embed.weight is restored.unembed.weight
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
        restored_sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=4, seed=11)
        load_distributed_checkpoint(checkpoint, restored, restored_optimizer, restored_sampler,
                                    expected_config=settings, expected_manifest_hash=manifest_hash(manifest))
        identical(normal.state_dict(), restored.state_dict())
        identical(optimizer.state_dict(), restored_optimizer.state_dict())
        identical(dict(normal.named_buffers()), dict(restored.named_buffers()))
        assert sampler.state_dict() == restored_sampler.state_dict()
        normal.eval(); restored.eval()
        probe = torch.tensor([[97, 98, 99, 100]])
        with torch.no_grad():
            identical(normal(probe), restored(probe))
        batch, restored_batch = sampler.next_batch(), restored_sampler.next_batch()
        identical(batch, restored_batch)
        update(normal, optimizer, batch)
        update(restored, restored_optimizer, restored_batch)
        identical(normal.state_dict(), restored.state_dict())
        identical(optimizer.state_dict(), restored_optimizer.state_dict())
        identical(dict(normal.named_buffers()), dict(restored.named_buffers()))
        if rank == 0:
            print(f'{case}: meta-only allocation and exact state/forward/update replay', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
