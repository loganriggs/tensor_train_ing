"""FSDP2 sharding for multi-GPU scale-up (brief §7, R31).

Each Block is one FSDP unit and the root holds the embedding, unembedding and final norm.
Parameters stay FP32 sharded masters. With `bf16_params=True` FSDP all-gathers BF16 copies
(MixedPrecisionPolicy). That is a separately verified numerical change, so the default
keeps FP32 parameters under BF16 autocast like the single-GPU reference.

Loss normalization contract: gradient reduction is a pure SUM across ranks (divide factor 1).
Each rank must back-propagate `local_loss_sum / global_valid_targets`, where
global_valid_targets is all-reduced over ranks and microbatches before the step. The result
equals the single-device update for the same global batch.

Replicated state: lagged-norm buffers are all-reduced in `LaggedRMS.commit`. Spectral u/v
are updated identically on every rank from the full matrix in `spectral_step`.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def shard_model(model, mesh=None, bf16_params: bool = False, reshard_after_forward: bool = True):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if bf16_params else MixedPrecisionPolicy()
    for blk in model.blocks:
        fully_shard(blk, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    for m in [model, *model.blocks]:
        m.set_gradient_divide_factor(1.0)
    return model


def global_count(n: torch.Tensor) -> torch.Tensor:
    """All-reduce a valid-target count (int64) across ranks."""
    n = n.clone().to(torch.int64)
    if dist.is_initialized():
        dist.all_reduce(n)
    return n


def full_state_dict(model) -> dict:
    """Gather an unsharded state dict (all ranks participate; rank 0 keeps CPU copies)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
