"""Flag finished training stages that read more input tokens than their training split holds (silent epoch wrap), and report the
input/trained token ratio. Run in every code-health review (like find_orphan_states.py).

    python scripts/check_data_epochs.py [--runs runs] [--ratio-warn 1.1]

Background (docs/redteam_2026-10-19.md A1): the trainer fills each update's target quota with whole micro-batches and discards the rest of
the last one, so an 8192 quota with 16x512 micro-batches read 1.96x the tokens trained; PackedSampler then silently starts epoch 2.
Exit code 1 if any stage wrapped past one epoch.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scan(runs: Path, ratio_warn: float):
    avail_cache, wrapped, high_ratio, total = {}, collections.defaultdict(list), collections.Counter(), 0
    for f in runs.rglob("result.json"):
        try:
            r = json.load(open(f)); p = r["progress"]; manifest = r["training_descriptor"]["configuration"]["data"]["manifest"]
            seen, trained = int(p["input_tokens_seen"]), int(p["tokens"])
        except (OSError, KeyError, ValueError, TypeError):
            continue
        if manifest not in avail_cache:
            try:
                avail_cache[manifest] = int(json.load(open(manifest))["tokens"]["train"])
            except (OSError, KeyError, ValueError):
                avail_cache[manifest] = None
        avail = avail_cache[manifest]
        if not avail:
            continue
        total += 1
        exp = f.relative_to(runs).parts[0]
        if seen > avail:
            wrapped[exp].append((str(f.parent.relative_to(runs)), seen / avail))
        if trained and seen / trained > ratio_warn:
            high_ratio[exp] += 1
    return total, wrapped, high_ratio


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--ratio-warn", type=float, default=1.1)
    a = ap.parse_args(argv)
    total, wrapped, high = scan(Path(a.runs), a.ratio_warn)
    print(f"stages checked: {total}")
    for exp, v in sorted(wrapped.items()):
        print(f"WRAPPED {exp}: {len(v)} stages read past one epoch (max {max(x[1] for x in v):.2f} epochs), e.g. {v[0][0]}")
    for exp, n in sorted(high.items()):
        print(f"ratio>{a.ratio_warn:g} {exp}: {n} stages read >{a.ratio_warn:g}x the tokens they trained on")
    return 1 if wrapped else 0


if __name__ == "__main__":
    sys.exit(main())
