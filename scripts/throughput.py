"""Steady-state training throughput (tokens/s) of one or more runs, from their metrics.jsonl.

    python scripts/throughput.py RUN_DIR_OR_METRICS [...]

Uses the second half of the logged rows (after compile and warm-up). For each argument, a directory is searched for every
`**/parent/metrics.jsonl` (the trained parent; branch rows are cooldown continuations). Used to catch throughput regressions
(2026-10-19: a 9% spectral_rms slowdown showed only here).
"""
import json
import sys
from pathlib import Path


def steady_rate(path) -> float | None:
    rows = [r for r in map(json.loads, open(path)) if "run_elapsed_seconds" in r and "tokens" in r]
    if len(rows) < 4:
        return None
    a, b = rows[len(rows) // 2], rows[-1]
    dt = b["run_elapsed_seconds"] - a["run_elapsed_seconds"]
    return (b["tokens"] - a["tokens"]) / dt if dt > 0 else None


def main(argv):
    for arg in argv:
        p = Path(arg)
        files = [p] if p.is_file() else sorted(p.rglob("parent/metrics.jsonl"))
        for f in files:
            r = steady_rate(f)
            name = f.parent.parent.name if f.parent.name == "parent" else f.parent.name
            print(f"{name:45s} {'n/a' if r is None else f'{r:8.0f} tok/s'}")


if __name__ == "__main__":
    main(sys.argv[1:])
