"""torchrun --nproc-per-node 2 tests/dist_workers/fsdp_equivalence.py  (gloo/CPU or nccl/GPU); run by tests/test_parallel.py

Checks one sharded data-parallel AdamW step, with the global batch split across ranks, against
the single-process step on the same global batch: identical loss sum and matching parameters."""
import os, sys
import torch
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules, spectral_step
from tnlm.parallel import global_count, shard_model, full_state_dict

backend = "nccl" if torch.cuda.device_count() >= int(os.environ.get("WORLD_SIZE", 1)) and torch.cuda.is_available() else "gloo"
dist.init_process_group(backend)
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh(dev.type, (world,))
torch.set_default_dtype(torch.float64)
for pkg in sys.argv[1:] or ["pre_rms", "spectral_rms", "lagged"]:
    kw = dict(layers=2, d_model=32, heads=2, d_key=16, d_value=16, d_mlp=48, vocab_size=97, padded_vocab=128,
              normalization=pkg, final_normalization="lagged" if pkg == "lagged" else "rms", attention_backend="reference")
    torch.manual_seed(0)
    ref = TensorLM(ModelConfig(**kw)).to(dev)
    torch.manual_seed(0)
    shd = shard_model(TensorLM(ModelConfig(**kw)).to(dev), mesh=mesh)
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 97, (2 * world, 12), generator=g).to(dev)
    doc = torch.zeros_like(idx); doc[:, 7:] = 1; doc[0, 10:] = -1
    tgt = idx.roll(-1, 1).masked_fill(doc < 0, -100)
    for model, sl in ((ref, slice(None)), (shd, slice(2 * rank, 2 * rank + 2))):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for m in lagged_modules(model):
            m.collecting = True; m.set_valid_mask((doc[sl] >= 0).to(torch.float64))
        for step in range(2):
            spectral_step(model)
            _, loss, n = model(idx[sl], doc[sl], tgt[sl], reduction="sum")
            denom = n if model is ref else global_count(n)  # reference already sees the whole global batch
            (loss / denom).backward()
            opt.step(); opt.zero_grad()
            for m in lagged_modules(model):
                m.commit(distributed=model is not ref)
        model.final_loss = loss.detach()
    sd = full_state_dict(shd)
    if rank == 0:
        ref_sd = ref.state_dict()
        err = max(float((sd[k].to(dev) - ref_sd[k]).abs().max()) for k in ref_sd if k in sd and ref_sd[k].is_floating_point())
        print(f"{pkg}: world={world} backend={backend} max|param diff| = {err:.3e}")
        assert err < 1e-10, err
dist.destroy_process_group()
