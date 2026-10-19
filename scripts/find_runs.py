"""List evaluated runs matching descriptor filters, best CE first (for picking like-for-like comparators).

    python scripts/find_runs.py width=512 depth=12 seed=202 optimizer=adamw rung_total_tokens=50003968 cooldown_status=cooled [--top 10]

Each KEY=VALUE is compared (as a string) with `training_descriptor[KEY]` of every runs/**/evaluation/evaluation.json; the candidate directory
name, CE and the descriptor's lr / normalization / optimizer are printed. Runs evaluated more than once are listed once (latest file).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find(runs: Path, where: dict) -> list[dict]:
    latest = {}
    for f in runs.rglob("evaluation/evaluation.json"):
        d = json.load(open(f)); td = d.get("training_descriptor") or {}
        if all(str(td.get(k)) == v for k, v in where.items()):
            key = (f.parts[len(runs.parts)], f.parent.parent.name)  # (experiment, candidate)
            if key not in latest or f.stat().st_mtime > latest[key]["mtime"]:
                latest[key] = {"experiment": key[0], "candidate": key[1], "ce": d["ce"], "lr": td.get("lr"),
                               "normalization": td.get("normalization"), "optimizer": td.get("optimizer"), "mtime": f.stat().st_mtime}
    return sorted(latest.values(), key=lambda r: r["ce"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*", help="KEY=VALUE descriptor filters")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)
    where = dict(f.split("=", 1) for f in a.filters)
    for r in find(Path(a.runs), where)[: a.top]:
        print(f"{r['ce']:.4f}  {r['experiment']}/{r['candidate']}  lr={r['lr']} {r['normalization']} {r['optimizer']}")


if __name__ == "__main__":
    main()
