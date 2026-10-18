"""List checkpoints (state.pt) under runs/ whose run already has a recorded evaluation: candidates for deletion.

    python scripts/find_orphan_states.py [--min-age-min 30] [--keep PATH_SUBSTRING ...]

A state counts as "evaluated" if its own run directory (≤3 levels above it) holds evaluation/evaluation.json or <run>.eval.json. States newer than
--min-age-min (a running job may still need them) and paths containing any --keep substring are excluded. Prints sizes and a total;
deletes nothing. Added after a branch-based experiment script left 36 GB of evaluated states on disk for 2 h (2026-10-18).
"""
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluated(state: Path, runs: Path) -> bool:
    """True if one of the state's own run directories (up to 3 levels above final/) holds its evaluation. Checks only direct children,
    never a recursive search: a sibling run's evaluation must not mark this state as done."""
    d = state.parent
    for _ in range(3):
        d = d.parent
        if not str(d).startswith(str(runs)) or d == runs:
            break
        if (d / "evaluation" / "evaluation.json").exists() or (d / "evaluation.json").exists() or (d.parent / f"{d.name}.eval.json").exists():
            return True
    return False


def orphans(runs: Path, min_age_min: float = 30, keep=()):
    now = time.time()
    for st in sorted(runs.rglob("state.pt")):
        if (now - st.stat().st_mtime) / 60 < min_age_min or any(k in str(st) for k in keep):
            continue
        if evaluated(st, runs):
            yield st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age-min", type=float, default=30); ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    total = 0
    for st in orphans(Path(a.runs), a.min_age_min, a.keep):
        size = st.stat().st_size; total += size
        print(f"{size / 1e9:7.2f} GB  {st.relative_to(ROOT) if str(st).startswith(str(ROOT)) else st}")
    print(f"TOTAL {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
