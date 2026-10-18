"""Tabulate the analysis cost of each normalization package from `tnlm analyze` outputs.

    python scripts/norm_cost_table.py LABEL=path/to/analysis.json [LABEL=path/to/analysis.json ...]

"Cost" = distance between the real model and the weight-defined polynomial an analysis would study (see docs/h100_readiness.md and the
2026-10-18 discussion). Five metrics, from structural to behavioural:
  1. sites        number of input-dependent normalizer sites (0 for a class-P model)
  2. denom spread median over sites of the q99/q01 ratio of each site's per-token denominator (1.0 = effectively constant)
  3. radial       median over sites of the median |radial Jacobian term| / |direct term| for random directions (hidden coupling)
  4. frozen gap   CE(all denominators frozen at calibrated constants) - CE(live); `inf` if frozen logits are non-finite
  5. ablation     median over components of |dCE_live - dCE_clamped| (how much norms change a weights-only importance ranking)
"""
import json
import math
import statistics
import sys


def _med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def row(path: str) -> dict:
    a = json.load(open(path))
    sites = a.get("normalizer_sensitivity", {}).get("sites", {}) or {}
    dyn = [s for s, d in sites.items() if "denominator" in d]
    spread = _med([d["denominator"]["q99"] / d["denominator"]["q01"] for d in sites.values()
                   if d.get("denominator") and d["denominator"].get("q01")])
    radial = _med([d["radial_over_direct_random_delta"]["q50"] for d in sites.values() if d.get("radial_over_direct_random_delta")])
    fd = a.get("fixed_denominator", {}) or {}
    if fd.get("ce_fixed") is not None and fd.get("ce_live") is not None:
        gap = fd["ce_fixed"] - fd["ce_live"]
    elif "ce_fixed" in (fd.get("nonfinite_metrics") or []):
        gap = math.inf
    else:
        gap = 0.0 if a.get("operator_class") == "P" else None
    comps = (a.get("ablation") or {}).get("components", {}) or {}
    abl = _med([abs(c["norm_mediated(live-clamped)"]["mean"]) for c in comps.values()
                if (c.get("norm_mediated(live-clamped)") or {}).get("mean") is not None])
    return {"class": a.get("operator_class"), "sites": len(dyn), "spread": spread, "radial": radial, "frozen_gap": gap, "ablation": abl}


def fmt(v, spec):
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return format(v, spec)


def main(argv):
    print("| package | class | sites | denom spread q99/q01 | radial/direct | frozen-denominator CE gap | ablation distortion |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for spec in argv:
        label, _, path = spec.partition("=")
        r = row(path)
        print(f"| {label} | {r['class']} | {r['sites']} | {fmt(r['spread'], '.3f')} | {fmt(r['radial'], '.4f')} | "
              f"{fmt(r['frozen_gap'], '.3f')} | {fmt(r['ablation'], '.3f')} |")


if __name__ == "__main__":
    main(sys.argv[1:])
