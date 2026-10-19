"""Scaling-law grid driver: one constant-LR parent per model size, cooled branches at several token budgets.

    python scripts/scaling_grid.py SIZE [--lr-mult M] [--tpp 5,10,20,40] [--device cuda] [--dry-run]
    python scripts/scaling_grid.py --plan                      # every size: params, tokens, GPU-hours, data needed

SIZE is a name from configs/scaling_grid/sizes.yaml (s38m ... s304m). For tokens-per-parameter budgets r (TOTAL params N, Chinchilla convention;
non-embedding counts are recorded too), D_r = r*N rounded to the update batch. The parent trains at constant LR to 0.9*max(D) and saves a milestone
at 0.9*D_r for every budget; each budget then gets a `tnlm cooldown` branch (linear decay over its last 10%, ending exactly at D_r) and a
`tnlm evaluate`. Sharing one parent costs max(D)*(1 + ~0.1 per budget) tokens instead of sum(D). Caveat: one LR per size across budgets; LRs in
sizes.yaml are set for the geometric-middle budget (horizon rule LR ~ D^-0.10 => <=1.23x off at the extremes, <=~0.01 CE on measured optima).

Idempotent and resumable at stage granularity: a stage with a finished `final/` (or evaluation.json) is skipped. Model states are pruned as soon as
their consumers are done (cooled finals after evaluation; milestones after their cooldown; the parent final at the end), per weights-not-needed.
Layout (find_runs.py / export_results.py compatible): runs/scaling_grid/<SIZE>[_lrxM]/{config.yaml, parent/, <SIZE>[_lrxM]_tpp<r>/{cooldown,evaluation}}.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "configs/scaling_grid"
TOK_S = {256: 100e3, 384: 85e3, 512: 66e3, 768: 45e3, 1024: 32e3, 1536: 17e3}  # measured medians at 512/1024 (find_runs --throughput); rest interpolated


def param_counts(model: dict) -> tuple[int, int]:
    import torch
    from tnlm.config import ModelConfig
    from tnlm.model import TensorLM
    keys = {k: v for k, v in model.items() if k in ModelConfig.__dataclass_fields__}
    with torch.device("meta"):
        m = TensorLM(ModelConfig(**keys))
    total = sum(p.numel() for p in m.parameters())
    emb = m.embed.weight.numel() + (0 if m.unembed.weight is m.embed.weight else m.unembed.weight.numel())
    return total, total - emb


def build(size: str, lr_mult: float, tpps: list[float], grid: Path = GRID, runs: Path = ROOT / "runs") -> dict:
    base = yaml.safe_load(open(grid / "base.yaml"))
    spec = yaml.safe_load(open(grid / "sizes.yaml"))[size]
    cfg = json.loads(json.dumps(base))
    cfg["model"].update(spec["model"])
    cfg["data"]["manifest"] = spec["data_manifest"]
    cfg["train"]["lr"] = spec["lr"] * lr_mult
    total, nonemb = param_counts(cfg["model"])
    batch = cfg["train"]["global_prediction_tokens_per_update"]
    budgets = sorted({int(round(r * total / batch)) * batch for r in tpps})
    branch = {d: ((9 * d // 10 + batch - 1) // batch) * batch for d in budgets}
    if any(branch[d] >= d for d in budgets):
        raise SystemExit(f"budget too small for a 10% cooldown at batch {batch}: {budgets}")
    cfg["train"]["rung_total_tokens"] = budgets[-1]
    cfg["project"]["milestone_tokens"] = sorted({9 * d // 10 for d in budgets})
    tag = size if lr_mult == 1 else f"{size}_lrx{lr_mult:g}"
    out = runs / "scaling_grid" / tag
    cfg["project"]["spend_ledger"] = str(out / "spend.json")
    return {"tag": tag, "out": out, "cfg": cfg, "total": total, "nonemb": nonemb, "batch": batch, "budgets": budgets, "branch": branch,
            "tpps": {d: d / total for d in budgets}, "width": cfg["model"]["d_model"]}


def plan_line(b: dict) -> str:
    tokens = branch_max = b["branch"][b["budgets"][-1]]
    tokens += sum(d - b["branch"][d] for d in b["budgets"])
    hours = tokens / TOK_S.get(b["width"], 30e3) / 3600
    return (f"{b['tag']:14s} N={b['total'] / 1e6:6.1f}M (non-emb {b['nonemb'] / 1e6:6.1f}M)  budgets "
            + ",".join(f"{d / 1e9:.2f}B" for d in b["budgets"])
            + f"  parent {branch_max / 1e9:.2f}B  train tokens {tokens / 1e9:.2f}B  ~{hours:.1f} GPU-h  data>= {b['budgets'][-1] / 1e9:.2f}B")


def sh(args: list[str], log: Path) -> int:
    with open(log, "a") as f:
        return subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def drop_state(ckpt: Path):
    for p in ckpt.rglob("state.pt"):
        p.unlink()


def run(b: dict, device: str, max_seconds: float) -> int:
    out, batch = b["out"], b["batch"]
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.yaml"
    if cfg_path.exists() and yaml.safe_load(open(cfg_path)) != b["cfg"]:
        raise SystemExit(f"{cfg_path} exists with a different configuration; refusing to mix (move it aside to restart)")
    cfg_path.write_text(yaml.safe_dump(b["cfg"], sort_keys=False))
    cli = [sys.executable, "-m", "tnlm.cli"]
    parent = out / "parent"
    last = b["branch"][b["budgets"][-1]]
    if not (parent / "final").exists():
        print(f"STAGE_START {b['tag']}/parent", flush=True)
        rc = sh(cli + ["train", "--config", str(cfg_path), "--run-dir", str(parent), "--max-steps", str(last // batch),
                       "--max-seconds", str(max_seconds), "--device", device], out / "parent.log")
        if rc:
            print(f"STAGE_FAILED {b['tag']}/parent rc={rc}", flush=True)
            return rc
    for d in b["budgets"]:
        name = f"{b['tag']}_tpp{round(b['tpps'][d], 3):g}"  # D is rounded to the batch, so 4.99997 -> tpp5
        cand = out / name
        ev = cand / "evaluation" / "evaluation.json"
        if ev.exists():
            continue
        steps = (d - b["branch"][d] + batch - 1) // batch
        milestone = parent / f"milestone-{9 * d // 10:012d}"
        resume = milestone if milestone.exists() else parent / "final"  # the largest budget's branch point is the parent's endpoint
        cand.mkdir(exist_ok=True)
        if not (cand / "cooldown" / "final").exists():
            print(f"STAGE_START {name}/cooldown D={d}", flush=True)
            rc = sh(cli + ["cooldown", "--config", str(cfg_path), "--resume", str(resume), "--cooldown-steps", str(steps),
                           "--cooldown-end-tokens", str(d), "--run-dir", str(cand / "cooldown"), "--max-steps", str(steps),
                           "--max-seconds", str(max_seconds), "--device", device], cand / "cooldown.log")
            if rc:
                print(f"STAGE_FAILED {name}/cooldown rc={rc}", flush=True)
                return rc
        if milestone.exists():
            drop_state(milestone)
        print(f"STAGE_START {name}/evaluation", flush=True)
        rc = sh(cli + ["evaluate", "--config", str(cfg_path), "--checkpoint", str(cand / "cooldown" / "final"), "--out", str(cand / "evaluation"),
                       "--max-seconds", "3600", "--device", device], cand / "evaluation.log")
        if rc:
            print(f"STAGE_FAILED {name}/evaluation rc={rc}", flush=True)
            return rc
        drop_state(cand / "cooldown")
        print(f"RUN_END {name} D={d} tpp={b['tpps'][d]:.2f} ce={json.load(open(ev))['ce']:.4f}", flush=True)
    drop_state(parent)
    for rec in parent.glob("recovery-*"):
        shutil.rmtree(rec, ignore_errors=True)
    print(f"GRID_DONE {b['tag']}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("size", nargs="?")
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--tpp", default="5,10,20,40", help="comma-separated tokens-per-parameter budgets (total params)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seconds", type=float, default=7 * 86400)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print every size's plan (default budgets) and exit")
    ap.add_argument("--grid-dir", default=str(GRID), help="directory with base.yaml + sizes.yaml (tests use a tiny one)")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    tpps = [float(x) for x in a.tpp.split(",")]
    if a.plan:
        for s in yaml.safe_load(open(Path(a.grid_dir) / "sizes.yaml")):
            print(plan_line(build(s, 1.0, tpps, Path(a.grid_dir), Path(a.runs_dir))))
        return 0
    b = build(a.size, a.lr_mult, tpps, Path(a.grid_dir), Path(a.runs_dir))
    print(plan_line(b), flush=True)
    return 0 if a.dry_run else run(b, a.device, a.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
