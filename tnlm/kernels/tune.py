"""Tile/stage/warp sweep for the polynomial attention kernel on the current GPU (BF16 fwd+bwd).

    python -m tnlm.kernels.tune --head-dim 64 --batch 4 --heads 16 --context 2048 --doc-len 300

Prints the fastest (block, num_stages, num_warps) for this GPU and head dim. Put the result in
`poly_attn.TUNING[(True, head_dim)]` (or the STAGES table) for that hardware.
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch

from ..attention import doc_positions
from . import poly_attn as pa


def _bench(f, n=10):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def sweep(head_dim=64, batch=4, heads=16, context=2048, doc_len=300, kind="signed_product",
          blocks=(32, 64, 128), stages=(1, 2, 3), warps=(4, 8)) -> list[dict]:
    mk = lambda: torch.randn(batch, context, heads, head_dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    q1, k1, q2, k2, v = (mk() for _ in range(5))
    docs = (torch.arange(context, device="cuda") // doc_len).expand(batch, -1).contiguous()
    scale = (doc_positions(docs) + 1).float().pow(-0.5)
    go = torch.randn(batch, heads, context, head_dim, device="cuda", dtype=torch.bfloat16)
    key, saved = (True, head_dim), dict(pa.TUNING)
    rows = []
    try:
        for blk, st, w in itertools.product(blocks, stages, warps):
            pa.TUNING[key] = (blk, st, w)
            try:
                ms = _bench(lambda: pa.poly_attention_triton(kind, q1, k1, q2, k2, v, docs, scale).backward(go))
            except Exception as ex:  # out of shared memory / registers for this config
                ms = float("inf")
            rows.append({"block": blk, "num_stages": st, "num_warps": w, "ms": ms})
            torch.cuda.empty_cache()
    finally:
        pa.TUNING.clear(); pa.TUNING.update(saved)
    return sorted(rows, key=lambda r: r["ms"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, d in (("head-dim", 64), ("batch", 4), ("heads", 16), ("context", 2048), ("doc-len", 300)):
        p.add_argument(f"--{a}", type=int, default=d)
    a = p.parse_args()
    rows = sweep(a.head_dim, a.batch, a.heads, a.context, a.doc_len)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "head_dim": a.head_dim, "best": rows[0], "top5": rows[:5]}, indent=1))
