"""Set a grid size's LR in configs/scaling_grid/sizes.yaml from its LR-check runs, before its main parent starts (queue step).

    python scripts/apply_grid_lr.py SIZE [--tpp 10] [--fallback-mult 0.36] [--dry-run]

Collects SIZE's evaluated check runs (runs/scaling_grid/<SIZE>_lrx*/<SIZE>_lrx*_tpp<r>/ and, if present, <SIZE>/<SIZE>_tpp<r>/), reading each LR from
its training descriptor. It fits CE = a·log(lr)² + b·log(lr) + c (scripts/fit_lr_optimum.fit). If the optimum is interior and the sweep is not flat
(CE range ≥ fit_lr_optimum.FLAT_CE) and the best measured check is not at an edge, that optimum is written; if the best check is at an
edge, its LR is written and NEEDS_BRACKET is printed; otherwise the current LR × --fallback-mult (default: the s38m measured ratio 0.36) is
written. The old LR, the new LR and the reason go into a YAML comment and stdout. Refuses if SIZE's main run directory already exists, so it never
changes the LR of a run in progress.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_lr_optimum as flo  # noqa: E402

SIZES = ROOT / "configs/scaling_grid/sizes.yaml"


def checks(runs: Path, size: str, tpp: str) -> dict:
    pts = {}
    for f in sorted(runs.glob(f"{size}*/{size}*_tpp{tpp}/evaluation/evaluation.json")):
        if not re.fullmatch(rf"{size}(_lrx[0-9.]+)?_tpp{tpp}", f.parent.parent.name):
            continue
        d = json.load(open(f))
        pts[float(d["training_descriptor"]["lr"])] = float(d["ce"])
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("size")
    ap.add_argument("--tpp", default="10")
    ap.add_argument("--fallback-mult", type=float, default=0.36)
    ap.add_argument("--runs", default=str(ROOT / "runs/scaling_grid"))
    ap.add_argument("--sizes", default=str(SIZES))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    runs, sizes = Path(a.runs), Path(a.sizes)
    if (runs / a.size).exists():
        print(f"REFUSED: {runs / a.size} exists (main run started); its LR must not change"); return 1
    text = sizes.read_text()
    m = re.search(rf"^{a.size}:\n(?:  .*\n)*?  lr: ([0-9.e-]+)", text, re.M)
    if not m:
        print(f"REFUSED: no lr for {a.size} in {sizes}"); return 1
    old = float(m.group(1))
    pts = checks(runs, a.size, a.tpp)
    r = flo.fit({0: pts})[0] if len(pts) >= 3 else {"opt": None, "interior": False, "ce": list(pts.values())}
    spread = (max(r["ce"]) - min(r["ce"])) if r["ce"] else 0.0
    best_lr = min(pts, key=pts.get) if pts else None
    lrs = sorted(pts)
    at_edge = flo.suggest_next({"lrs": lrs, "ce": [pts[x] for x in lrs]}) is not None  # same edge rule as fit_lr_optimum NEXT hints
    if at_edge and spread >= flo.FLAT_CE:
        # the best measured point is at the edge of the checks: a quadratic minimum hugging the edge is not trustworthy (s67m, 2026-10-20)
        new, why = best_lr, f"best check at the EDGE ({best_lr:.3g}); used it, NEEDS_BRACKET one x0.5 step further"
    elif r["opt"] and r["interior"] and spread >= flo.FLAT_CE:
        new, why = r["opt"], f"fitted interior optimum from {len(pts)} tpp{a.tpp} checks (CE spread {spread:.3f})"
    else:
        new, why = old * a.fallback_mult, f"fallback x{a.fallback_mult} (checks: {len(pts)} points, interior={r['interior']}, spread {spread:.3f})"
    print(f"{a.size}: lr {old:.3g} -> {new:.3g} ({why}); points " + ", ".join(f"{lr:.3g}:{ce:.4f}" for lr, ce in sorted(pts.items())))
    if not a.dry_run:
        start, end = m.span(1)
        line_end = text.index("\n", end)
        text = text[:start] + f"{new:.4g}   # apply_grid_lr.py: {why}; was {old:.3g}" + text[line_end:]
        sizes.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
