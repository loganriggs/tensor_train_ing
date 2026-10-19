"""Fit a Chinchilla scaling law L(N, D) = E + A/N^alpha + B/D^beta to the scaling-grid evaluations, and derive compute-optimal allocations.

    python scripts/fit_scaling_law.py [--params total|nonemb] [--compute 2.4e23 ...] [--runs runs]

Points: every runs/scaling_grid/<SIZE>/<SIZE>_tpp<r>/evaluation/evaluation.json of the main sizes (LR-check tags `_lrx*` are excluded),
with N from the size's config.yaml (total or non-embedding parameters) and D = training_descriptor.actual_prediction_tokens (each cooled branch's
own endpoint; rung_total_tokens is the parent's maximum), L = held-out CE.

Fit (Hoffmann et al. 2022, Approach 3): minimize the Huber loss (delta 1e-3) of log L - LSE(a - alpha log N, b - beta log D, e) with L-BFGS from a
grid of initializations; A = e^a, B = e^b, E = e^e. With C = 6 N D, the loss-optimal allocation is N_opt = G (C/6)^(beta/(alpha+beta)),
D_opt = (C/6) / N_opt, G = (alpha A / (beta B))^(1/(alpha+beta)), so tokens/param at the optimum varies with C unless alpha = beta.
torch only (no scipy in this environment).
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def collect(runs: Path, params: str = "total") -> list[tuple[float, float, float, str]]:
    from scaling_grid import param_counts
    pts = []
    for size_dir in sorted((runs / "scaling_grid").glob("*")):
        if "_lrx" in size_dir.name or not (size_dir / "config.yaml").exists():
            continue
        total, nonemb = param_counts(yaml.safe_load(open(size_dir / "config.yaml"))["model"])
        n = total if params == "total" else nonemb
        for ev in sorted(size_dir.glob("*/evaluation/evaluation.json")):
            d = json.load(open(ev))
            pts.append((float(n), float(d["training_descriptor"]["actual_prediction_tokens"]), float(d["ce"]), ev.parent.parent.name))
    return pts


def fit(points, delta: float = 1e-3) -> dict:
    """points: iterable of (N, D, L). Returns {A, B, E, alpha, beta, loss}."""
    N, D, L = (torch.tensor([p[i] for p in points], dtype=torch.float64) for i in range(3))
    logN, logD, logL = N.log(), D.log(), L.log()
    best = None
    for a0, b0, e0, al0, be0 in itertools.product((0., 5., 10., 15.), (0., 5., 10., 15.), (-1., 0., 1., 1.5), (0., .5, 1.), (0., .5, 1.)):
        x = torch.tensor([a0, b0, e0, al0, be0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([x], lr=1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            a, b, e, al, be = x
            pred = torch.logsumexp(torch.stack([a - al * logN, b - be * logD, e.expand_as(logN)]), dim=0)
            loss = torch.nn.functional.huber_loss(pred, logL, delta=delta, reduction="sum")
            loss.backward()
            return loss
        try:
            opt.step(closure)  # NB: returns the FIRST closure loss, not the final one
            with torch.no_grad():
                a, b, e, al, be = x
                pred = torch.logsumexp(torch.stack([a - al * logN, b - be * logD, e.expand_as(logN)]), dim=0)
                loss = torch.nn.functional.huber_loss(pred, logL, delta=delta, reduction="sum").item()
        except RuntimeError:
            continue
        if math.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, x.detach().clone())
    loss, (a, b, e, al, be) = best[0], best[1].tolist()
    return {"A": math.exp(a), "B": math.exp(b), "E": math.exp(e), "alpha": al, "beta": be, "loss": loss}


def optimal(f: dict, compute: float) -> dict:
    al, be = f["alpha"], f["beta"]
    g = (al * f["A"] / (be * f["B"])) ** (1 / (al + be))
    n = g * (compute / 6) ** (be / (al + be))
    d = compute / 6 / n
    return {"compute": compute, "N_opt": n, "D_opt": d, "tokens_per_param": d / n,
            "L_opt": f["E"] + f["A"] / n ** al + f["B"] / d ** be}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", choices=("total", "nonemb"), default="total")
    ap.add_argument("--compute", type=float, nargs="*", default=[1e19, 1e20, 1e21, 2.4e23],
                    help="FLOPs (6ND) to report optimal allocations at; 2.4e23 = the planned 1.41B x 28B H100 run")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    pts = collect(Path(a.runs), a.params)
    sizes = sorted({p[0] for p in pts})
    print(f"{len(pts)} points over {len(sizes)} sizes ({a.params} params): " + ", ".join(f"{n / 1e6:.1f}M" for n in sizes))
    for n, d, l, name in pts:
        print(f"  {name:22s} N={n / 1e6:7.1f}M D={d / 1e9:6.2f}B tpp={d / n:6.1f} CE={l:.4f}")
    if len(sizes) < 3 or len(pts) < 6:
        print("too few sizes/points for a 5-parameter fit (need >= 3 sizes, >= 6 points)")
        return 1
    f = fit([p[:3] for p in pts])
    print(f"fit: L = {f['E']:.3f} + {f['A']:.4g}/N^{f['alpha']:.3f} + {f['B']:.4g}/D^{f['beta']:.3f}   (Huber loss {f['loss']:.3g})")
    for c in a.compute:
        o = optimal(f, c)
        print(f"  C={c:.2g}: N_opt={o['N_opt'] / 1e6:,.0f}M  D_opt={o['D_opt'] / 1e9:,.1f}B  tokens/param={o['tokens_per_param']:.1f}  L={o['L_opt']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
