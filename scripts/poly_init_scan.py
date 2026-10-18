"""Init-time scale scan for polynomial (lagged) models on real text: does any token's scale explode with depth?

    python scripts/poly_init_scan.py [--layers 12] [--d-model 256] [--heads 4] [--set residual_alpha=0.1 ...] [--set ...]

Each `--set` group (space-separated KEY=VALUE model fields) is one configuration; the bare default is always included. For each, it
builds a `lagged` model, calibrates every site in order on 256×64 tokens of the pilot calibration split (each site's constant = the mean
square of its input with earlier sites already calibrated), and prints the final-norm mean square and the worst per-token max/median
over sites. Healthy: both O(1). Exploding: ≫1 (the 2026-10-18 finding: default init gives ~10^19; residual_alpha ≤ 0.3 gives O(1)).
Use real text, not random ids: random tokens hid the outliers. Promoted from three ad-hoc scans on 2026-10-18.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.normalization import LaggedRMS
from tnlm.token_format import read_split_tokens

ROOT = Path(__file__).resolve().parents[1]


def scan(X, layers, d_model, heads, **kw):
    torch.manual_seed(0)
    dk = d_model // heads
    m = TensorLM(ModelConfig(layers=layers, d_model=d_model, heads=heads, d_key=dk, d_value=dk, d_mlp=3 * d_model, padded_vocab=50304,
                             normalization="lagged", final_normalization="lagged", attention_backend="reference", **kw)).eval()
    doc = torch.zeros_like(X)
    norms = [mod for mod in m.modules() if isinstance(mod, LaggedRMS)]
    worst = 0.0
    for n in norms:
        acts = {}
        h = n.register_forward_pre_hook(lambda mod, a: acts.__setitem__("x", a[0].detach()))
        with torch.no_grad():
            m(X, doc)
        h.remove()
        pt = acts["x"].double().pow(2).mean(-1)
        n.mean_sq.copy_(pt.mean()); n.calibrated.fill_(True)
        worst = max(worst, float(pt.max() / pt.median()))
    return float(norms[-1].mean_sq), worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--set", nargs="+", action="append", default=[])
    a = ap.parse_args(argv)
    tok = np.asarray(read_split_tokens(ROOT / "data/fineweb_pilot_5m", "calibration"))  # shared format (tnlm.token_format)
    X = torch.from_numpy(tok[: 256 * 64].astype(np.int64)).view(256, 64)
    for group in [[]] + a.set:
        kw = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in group)}
        ms, w = scan(X, a.layers, a.d_model, a.heads, **kw)
        print(f"{str(kw) or 'default':45s} final mean_sq={ms:10.3e}   worst per-token max/median={w:10.3e}")


if __name__ == "__main__":
    main()
