"""Print a compact table from a `tnlm compare_evaluations --pairs` output (comparisons.json).

    python scripts/print_pairs.py runs/confirmation/comparisons_s101/comparisons.json [--markdown]

Schema used: {"pairs": [{"name", "left", "right", "result": {"delta_ce_right_minus_left", "documents",
"uncertainty": {"delta_ce_95_ci": [lo, hi]}}}]}. CIs are nominal document-bootstrap intervals, not seed intervals.
"""
import json
import sys


def rows(path):
    d = json.load(open(path))
    for p in d["pairs"]:
        r = p["result"]
        lo, hi = r["uncertainty"]["delta_ce_95_ci"]
        yield p["name"], r["delta_ce_right_minus_left"], lo, hi, r["documents"]


def main(argv):
    path, md = argv[0], "--markdown" in argv
    if md:
        print("| Comparison | ΔCE (right − left) | 95% CI | docs |\n| --- | ---: | --- | ---: |")
    for name, d, lo, hi, n in rows(path):
        print(f"| {name} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {n} |" if md
              else f"{name:40s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  docs={n}")


if __name__ == "__main__":
    main(sys.argv[1:])
