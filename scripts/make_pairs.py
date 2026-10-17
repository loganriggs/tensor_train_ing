"""Write a `tnlm compare_evaluations --pairs` spec with verified absolute paths.

    python scripts/make_pairs.py OUT.yaml NAME=LEFT_RUN,RIGHT_RUN [NAME=LEFT_RUN,RIGHT_RUN ...]

A RUN is any directory under runs/ that contains exactly one `**/evaluation/evaluation.json`. The candidate's own directory is enough,
e.g. `width2048/v2/w2048ws_lr0.00135_s202`. Paths are resolved here and written absolute, and the script exits non-zero
(writing nothing) if any run has zero or several evaluations. Added after two hand-written pair specs had wrong paths on 2026-10-17
(a stale `v1/` prefix, then cwd-relative paths that the tool resolves relative to the spec file).
"""
import sys
from pathlib import Path

import yaml

RUNS = Path(__file__).resolve().parents[1] / "runs"


def resolve(run: str, runs: Path = RUNS) -> str:
    base = runs / run
    hits = sorted(base.glob("**/evaluation/evaluation.json")) if base.is_dir() else []
    if len(hits) != 1:
        raise SystemExit(f"{run}: expected exactly one evaluation.json under {base}, found {len(hits)}")
    return str(hits[0].resolve())


def main(argv, runs: Path = RUNS):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out, specs = Path(argv[0]), argv[1:]
    pairs = []
    for spec in specs:
        name, _, rest = spec.partition("=")
        left, _, right = rest.partition(",")
        if not (name and left and right):
            raise SystemExit(f"bad pair spec {spec!r}; want NAME=LEFT_RUN,RIGHT_RUN")
        pairs.append({"name": name, "left": resolve(left, runs), "right": resolve(right, runs)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(pairs, sort_keys=False))
    print(f"{out}: {len(pairs)} pairs, all inputs exist")


if __name__ == "__main__":
    main(sys.argv[1:])
