"""Coordinate check across width at fixed depth and head dimension (brief §5).

For each width, records RMS of residual streams, branch outputs and logits at init and after `steps`
optimizer steps at a fixed LR, plus the RMS of the change since init. Fits log-log slopes against width:
a slope near 0 means width-stable. Positive slopes in the per-step change mean the effective LR grows
with width, so an LR tuned at the screen width will not transfer unchanged. This is a diagnostic, not a
muP derivation, and the parameterization label stays "explicit conventional" unless the slopes support it.
"""
from __future__ import annotations

import math

import torch

from .config import ModelConfig, TrainConfig
from .model import TensorLM
from .optim import make_optimizer


def _rms(t):
    return float(t.detach().float().pow(2).mean().sqrt())


@torch.no_grad()
def _probe(model, idx):
    stats, hooks = {}, []
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: stats.__setitem__(f"resid{i}", o)))
        hooks.append(blk.attn.register_forward_hook(lambda m, a, o, i=i: stats.__setitem__(f"attn{i}", o)))
        hooks.append(blk.mlp.register_forward_hook(lambda m, a, o, i=i: stats.__setitem__(f"mlp{i}", o)))
    try:
        stats["logits"] = model(idx)
    finally:
        for h in hooks:
            h.remove()
    return {k: v.detach().float().clone() for k, v in stats.items()}


def _batch(g, batch, context, vocab, task):
    x = torch.randint(0, vocab, (batch, context), generator=g)
    if task == "repeat":  # learnable: second half copies the first, so gradients carry real feature-learning signal
        x[:, context // 2:] = x[:, : context - context // 2]
    return x


def run(widths=(256, 512, 1024), *, layers=4, head_dim=64, steps=3, lr=4.5e-4, optimizer="adamw",
        batch=8, context=256, vocab=8192, device="cuda", seed=0, task="random", train_kw=None, **model_kw) -> dict:
    """train_kw: extra TrainConfig fields (e.g. {"muon_adjust_lr": "spectral_unclamped"})."""
    rows = {}
    for w in widths:
        torch.manual_seed(seed)
        poly = model_kw.get("attention", "signed_product") != "softmax"
        d_mlp = int(round((4 * w - (2 * w / 3 if poly else 0)) / 64) * 64) if "d_mlp" not in model_kw else model_kw["d_mlp"]
        cfg = ModelConfig(layers=layers, d_model=w, heads=w // head_dim, d_key=head_dim, d_value=head_dim, d_mlp=d_mlp,
                          vocab_size=vocab, padded_vocab=vocab, **{k: v for k, v in model_kw.items() if k != "d_mlp"})
        m = TensorLM(cfg).to(device).train()
        opt = make_optimizer(m, TrainConfig(optimizer=optimizer, lr=lr, **(train_kw or {})))
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        probe = _batch(g, batch, context, vocab, task).to(device)
        init = _probe(m, probe)
        for _ in range(steps):
            idx = _batch(g, batch, context, vocab, task).to(device)
            with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
                _, loss = m(idx, targets=idx.roll(-1, 1))
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        after = _probe(m, probe)
        rows[w] = {k: {"init": _rms(init[k]), "after": _rms(after[k]), "delta": _rms(after[k] - init[k])} for k in init}
        del m, opt
    # log-log slopes vs width (needs >= 2 widths)
    xs = [math.log(w) for w in widths]
    slopes = {}
    for key in (rows[widths[0]] if len(widths) > 1 else []):
        for f in ("init", "delta"):
            ys = [math.log(max(rows[w][key][f], 1e-30)) for w in widths]
            xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
            slopes[f"{key}.{f}"] = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    return {"widths": list(widths), "layers": layers, "head_dim": head_dim, "optimizer": optimizer, "lr": lr, "task": task,
            "steps": steps, "rows": rows, "loglog_slopes": slopes}


def run_depth(depths=(2, 4, 8, 16), *, width=512, **kw) -> dict:
    """Same check across depth at fixed width/head dim: per-layer quantities are reported for the
    first and last block, and slopes are taken against depth."""
    rows = {}
    for L in depths:
        r = run(widths=(width,), layers=L, **kw)["rows"][width]
        rows[L] = {"resid_first": r["resid0"], "resid_last": r[f"resid{L - 1}"], "attn_last": r[f"attn{L - 1}"],
                   "mlp_last": r[f"mlp{L - 1}"], "logits": r["logits"]}
    xs = [math.log(L) for L in depths]
    slopes = {}
    for key in rows[depths[0]]:
        for f in ("init", "delta"):
            ys = [math.log(max(rows[L][key][f], 1e-30)) for L in depths]
            xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
            slopes[f"{key}.{f}"] = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    return {"depths": list(depths), "width": width, "rows": rows, "loglog_slopes": slopes}


if __name__ == "__main__":
    import json, sys
    opt = sys.argv[1] if len(sys.argv) > 1 else "adamw"
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 4.5e-4
    r = run(optimizer=opt, lr=lr)
    print(json.dumps({k: round(v, 2) for k, v in r["loglog_slopes"].items()}))
