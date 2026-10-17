"""Model-shape presets and a measured-constant memory planner (brief §7).

Polynomial presets are parameter-matched to a conventional SwiGLU+softmax control of the same
depth/width. Two extra QK projections cost 2*d^2 per layer, which comes out of the MLP:
d_mlp_poly ~= d_mlp_softmax - 2d/3, rounded to a multiple of 128. Head dim is 64 up to the 500M
confirmation model (fixed for width-transfer tests) and 128 from 1B up, as in conventional LLMs.
Parameter counts come from instantiating on the meta device, never from formulas.

Memory constants were MEASURED on an RTX 5000 Ada (torch 2.14, BF16 autocast, FP32 params,
Triton attention, chunked CE) at d<=1024, T=1024:
  activations without block checkpointing ~ 66 * d bytes / token / layer
  activations with    block checkpointing ~  8 * d bytes / token / layer (+ one block's full set)
Re-measure on the target GPU before trusting a tight fit. The planner is capacity planning, not a fit proof.
"""
from __future__ import annotations

import dataclasses

import torch

from .config import ModelConfig

ACT_BYTES_PER_D = {False: 66.0, True: 8.0}
STATE_BYTES_PER_PARAM = 16  # FP32 param + FP32 grad + 2 x FP32 Adam moments

# name: (layers, d_model, heads, head_dim, d_mlp_softmax)
_BASE = {
    "screen_100m": (12, 512, 8, 64, 2048),
    "confirm_500m": (24, 1024, 16, 64, 4096),
    "width_1024_12l": (12, 1024, 16, 64, 4096),  # isolated width-transfer partner of screen_100m
    "pilot_1b": (24, 2048, 16, 128, 5504),
    "pilot_3b": (32, 2560, 20, 128, 6912),
    "funded_7b": (32, 4096, 32, 128, 11008),
    "funded_32b": (64, 5120, 40, 128, 27648),
}
_EXACT_POLY_MLP = {"screen_100m": 1664, "confirm_500m": 3392, "width_1024_12l": 3392}


def preset(name: str, polynomial: bool = True, **overrides) -> ModelConfig:
    L, d, H, hd, m_soft = _BASE[name]
    if polynomial:
        m = _EXACT_POLY_MLP.get(name) or int(round((m_soft - 2 * d / 3) / 128) * 128)
        kw = dict(attention="signed_product", mlp="bilinear", d_mlp=m)
    else:
        kw = dict(attention="softmax", mlp="swiglu", d_mlp=m_soft)
    kw.update(layers=L, d_model=d, heads=H, d_key=hd, d_value=hd)
    kw.update(overrides)
    return ModelConfig(**kw)


def count_params(cfg: ModelConfig) -> dict:
    from .model import Block
    with torch.device("meta"):
        blk = Block(cfg)
    per_block = sum(p.numel() for p in blk.parameters())
    per_block_mat = sum(p.numel() for p in blk.parameters() if p.dim() >= 2)
    emb = cfg.padded_vocab * cfg.d_model * (1 if cfg.tie_embeddings else 2)
    return {"total": cfg.layers * per_block + emb + cfg.d_model, "matrices": cfg.layers * per_block_mat + emb,
            "non_embedding": cfg.layers * per_block_mat, "per_block": per_block}


def memory_plan(cfg: ModelConfig, *, gpus: int, micro_tokens_per_gpu: int, gpu_gb: float = 80.0,
                checkpointing: bool | None = None, workspace_gb: float = 4.0) -> dict:
    """Per-GPU bytes for FSDP-sharded states + activations of one microbatch."""
    ckpt = cfg.activation_checkpointing if checkpointing is None else checkpointing
    n = count_params(cfg)["total"]
    states = STATE_BYTES_PER_PARAM * n / gpus
    # FSDP transiently unshards one block (FP32) for forward/backward, plus prefetch of the next
    unshard = 2 * 4 * count_params(cfg)["per_block"]
    act = ACT_BYTES_PER_D[ckpt] * cfg.d_model * cfg.layers * micro_tokens_per_gpu
    if ckpt:  # recompute one block at a time with its full activations
        act += ACT_BYTES_PER_D[False] * cfg.d_model * micro_tokens_per_gpu
    loss = 4 * 4096 * cfg.padded_vocab * (2 if cfg.loss_chunk_tokens else micro_tokens_per_gpu / 4096)
    total = states + unshard + act + loss + workspace_gb * 1e9
    return {"params_B": round(n / 1e9, 3), "gpus": gpus, "micro_tokens_per_gpu": micro_tokens_per_gpu,
            "checkpointing": ckpt, "states_GB": round(states / 1e9, 1), "activations_GB": round(act / 1e9, 1),
            "unshard_GB": round(unshard / 1e9, 1), "loss_GB": round(loss / 1e9, 1),
            "total_GB": round(total / 1e9, 1), "fits": total < 0.92 * gpu_gb * 1e9}


def min_gpus(cfg: ModelConfig, micro_tokens_per_gpu: int, gpu_gb: float = 80.0, checkpointing: bool = True,
             max_gpus: int = 1024) -> dict | None:
    g = 1
    while g <= max_gpus:
        plan = memory_plan(cfg, gpus=g, micro_tokens_per_gpu=micro_tokens_per_gpu, gpu_gb=gpu_gb, checkpointing=checkpointing)
        if plan["fits"]:
            return plan
        g *= 2
    return None


def table(micro_tokens_per_gpu: int = 8192, gpu_gb: float = 80.0) -> list[dict]:
    rows = []
    for name in _BASE:
        for poly in (True, False):
            cfg = preset(name, poly, loss_chunk_tokens=4096)
            c = count_params(cfg)
            plan = min_gpus(cfg, micro_tokens_per_gpu, gpu_gb)
            rows.append({"preset": name, "family": "polynomial" if poly else "softmax-control",
                         "layers": cfg.layers, "d_model": cfg.d_model, "heads": cfg.heads, "head_dim": cfg.d_key,
                         "d_mlp": cfg.d_mlp, "params_B": round(c["matrices"] / 1e9, 3),
                         "min_80GB_gpus_ckpt": plan["gpus"] if plan else None,
                         "per_gpu_GB": plan["total_GB"] if plan else None})
    return rows


if __name__ == "__main__":
    import json
    for r in table():
        print(json.dumps(r))
