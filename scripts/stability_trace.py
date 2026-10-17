"""Summarize the periodic stability logs (`train.stability_every_steps`) of a training run.

    python scripts/stability_trace.py runs/width2048/v2/<cand>/<cand>/parent/metrics.jsonl [more metrics.jsonl ...]

Prints, per logged step, the worst-layer max |S1*S2|, the last-layer S1 RMS, and the worst MLP-product max and excess kurtosis.
Schema: metrics.jsonl records with `stability.stats.{per_layer, worst}` (see `tnlm.analysis.stability_stats`). A >10x jump in
the S1*S2 max between logs is the instability signal documented in docs/scaling.md. Steady runaways are flagged too:
> 10x the run's first logged value, or above ABS_LIMIT (healthy 50M-token runs stay below ~1e4; the unstable default 1B at
LR 1.35e-3 reached 2.7e5 by step 500 and 1.6e7 by step 4000).
"""
ABS_LIMIT = 1e5
import json
import sys


def trace(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        st = (r.get("stability") or {}).get("stats")
        if not st:
            continue
        w, last = st["worst"], st["per_layer"][-1]
        rows.append((r["step"], w.get("S1S2_max_abs"), last.get("S1_rms"), w.get("mlp_prod_max_abs"),
                     w.get("mlp_prod_excess_kurtosis"), r.get("loss")))
    return rows


def main(paths):
    for p in paths:
        print(p)
        print(f"{'step':>7} {'S1S2max':>10} {'lastS1rms':>9} {'mlpmax':>9} {'mlpkurt':>9} {'loss':>7}")
        prev = first = None
        for step, s12, s1, mm, mk, loss in trace(p):
            flags = []
            if prev and s12 and s12 > 10 * prev:
                flags.append(">10x jump")
            if first and s12 and s12 > 10 * first:
                flags.append(">10x first log")
            if s12 and s12 > ABS_LIMIT:
                flags.append(f">{ABS_LIMIT:.0e} abs")
            print(f"{step:7d} {s12:10.1f} {s1:9.2f} {mm:9.1f} {mk:9.1f} {loss:7.3f}" + ("  <-- " + ", ".join(flags) if flags else ""))
            prev = s12
            first = first or s12


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet when piped into head
    main(sys.argv[1:])
