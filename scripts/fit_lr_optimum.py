"""Fit the optimal LR per group from cooled evaluations, then a power law of the optimum across groups.

    python scripts/fit_lr_optimum.py [--pattern 'w(?P<group>\\d+)ws_lr(?P<lr>[0-9.e-]+)_s101'] [--runs runs]

Collects every runs/**/evaluation/evaluation.json whose candidate directory name matches --pattern (named groups `group` and `lr`),
keeps one CE per (group, lr) (the most recent file if duplicated), fits CE = a·log(lr)² + b·log(lr) + c per group with >=3 LRs, and reports
the fitted optimum and whether it lies inside the sampled LR range (an optimum outside the range is an extrapolation, flagged). With >=2
groups whose optimum is interior, it fits log(opt) = k + slope·log(group). Default pattern: width-scaled widths (the exponent fit, where a
slope of 0 means the width rule transfers the optimum exactly).
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def collect(runs: Path, pattern: str, also=()) -> dict:
    rxs = [re.compile(p) for p in (pattern, *also)]
    pts = {}
    for f in runs.rglob("evaluation/evaluation.json"):
        name = f.parent.parent.name
        m = next((r.fullmatch(name) for r in rxs if r.fullmatch(name)), None)
        if not m:
            continue
        g, lr = m.group("group"), float(m.group("lr"))
        g = int(g) if g.isdigit() else g  # widths are numeric; optimizer/variant names are fine too
        prev = pts.get((g, lr))
        if prev is None or f.stat().st_mtime > prev[1]:
            pts[(g, lr)] = (json.load(open(f))["ce"], f.stat().st_mtime)
    out = {}
    for (g, lr), (ce, _) in pts.items():
        out.setdefault(g, {})[lr] = ce
    return out


def fit(points: dict) -> dict:
    res = {}
    for g, d in sorted(points.items()):
        lrs = sorted(d)
        if len(lrs) < 3:
            res[g] = {"lrs": lrs, "ce": [d[x] for x in lrs], "opt": None, "interior": False}
            continue
        x, y = np.log(lrs), np.array([d[v] for v in lrs])
        a, b, c = np.polyfit(x, y, 2)
        opt = math.exp(-b / (2 * a)) if a > 0 else None
        res[g] = {"lrs": lrs, "ce": list(y), "opt": opt, "min_ce": c - b * b / (4 * a) if a > 0 else None,
                  "interior": opt is not None and lrs[0] <= opt <= lrs[-1]}
    return res


def suggest_next(r: dict, factor: float = 2.0):
    """If the lowest sampled CE sits at the edge of the sampled LRs, the next LR to bracket it (one more factor-2 step outward); else None."""
    if not r["lrs"]:
        return None
    best = min(range(len(r["lrs"])), key=lambda i: r["ce"][i])
    if len(r["lrs"]) >= 2 and best == len(r["lrs"]) - 1:
        return r["lrs"][-1] * factor
    if len(r["lrs"]) >= 2 and best == 0:
        return r["lrs"][0] / factor
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=r"w(?P<group>\d+)ws_lr(?P<lr>[0-9.e-]+)_s101")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--also", nargs="*", default=[r"w(?P<group>512)_lr(?P<lr>[0-9.e-]+)_s101"],
                    help="extra patterns (default: plain w512 runs, identical to width-scaled at the 512 base width)")
    a = ap.parse_args(argv)
    res = fit(collect(Path(a.runs), a.pattern, a.also))
    for g, r in res.items():
        pts = ", ".join(f"{lr:g}:{ce:.3f}" for lr, ce in zip(r["lrs"], r["ce"]))
        o = "n/a (<3 LRs)" if r["opt"] is None and len(r["lrs"]) < 3 else (f"{r['opt']:.2e}" if r["opt"] else "no minimum")
        flag = "" if r["interior"] or r["opt"] is None else "  (outside sampled range: extrapolated)"
        print(f"group {g}: {pts}  ->  optimum {o}{flag}")
        nxt = suggest_next(r)
        if nxt:
            print(f"  NEXT group {g}: add lr {nxt:.3g} (best sampled point is at the edge; bracket before trusting the optimum)")
    good = [(g, r["opt"]) for g, r in res.items() if r["interior"] and isinstance(g, int)]
    if len(good) >= 2:
        slope, k = np.polyfit(np.log([g for g, _ in good]), np.log([o for _, o in good]), 1)
        print(f"optimum ∝ group^{slope:.3f}  (from {len(good)} groups with interior optima)")


if __name__ == "__main__":
    main()
