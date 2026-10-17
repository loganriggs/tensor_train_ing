"""Canonical component source to real FSDP shards, including uneven scalar shards."""
from datetime import timedelta
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.sharded_initialization import sharded_model_from_cpu_components


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=60))
    mesh = init_device_mesh('cpu', (dist.get_world_size(),))
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        for case in ('pre_rms', 'spectral_rms', 'lagged', 'tied', 'affine'):
            norm = case if case in ('pre_rms', 'spectral_rms', 'lagged') else 'pre_rms'
            cfg = ModelConfig(layers=2, d_model=8, heads=1, d_key=4, d_value=4,
                d_mlp=12, padded_vocab=50257, normalization=norm,
                final_normalization='lagged' if norm == 'lagged' else 'rms',
                tie_embeddings=case == 'tied', mlp='affine_bilinear' if case == 'affine' else 'bilinear',
                attention_backend='reference')
            torch.manual_seed(91)
            reference = TensorLM(cfg).eval()
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(91)
            model = sharded_model_from_cpu_components(cfg, mesh, 'cpu').eval()
            assert torch.equal(torch.get_rng_state(), expected_rng), case
            expected = dict(reference.named_parameters()) | dict(reference.named_buffers())
            for name, tensor in (dict(model.named_parameters()) | dict(model.named_buffers())).items():
                value = tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor
                assert torch.equal(value, expected[name]), (dtype, case, name)
            if cfg.tie_embeddings:
                assert model.embed.weight is model.unembed.weight
            idx = torch.tensor([[97, 98, 99, 100]])
            docs = torch.tensor([[0, 0, 1, 1]])
            with torch.no_grad():
                torch.testing.assert_close(model(idx, docs), reference(idx, docs), rtol=0, atol=0)
            del model, reference
    if dist.get_rank() == 0:
        print('canonical fresh sharding: all cases passed', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
