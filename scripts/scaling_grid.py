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
    """(total, non-embedding) parameters of a model section; delegates to tnlm.shapes.count_params (the repo's single counter)."""
    from tnlm.config import ModelConfig
    from tnlm.shapes import count_params
    c = count_params(ModelConfig(**{k: v for k, v in model.items() if k in ModelConfig.__dataclass_fields__}))
    return c["total"], c["non_embedding"]


def build(size: str, lr_mult: float, tpps: list[float], grid: Path = GRID, runs: Path = ROOT / "runs") -> dict:
    base = yaml.safe_load(open(grid / "base.yaml"))
    spec = yaml.safe_load(open(grid / "sizes.yaml"))[size]
    cfg = json.loads(json.dumps(base))
    cfg["model"].update(spec["model"])
    cfg["data"]["manifest"] = spec["data_manifest"]
    cfg["train"].update(spec.get("train", {}))  # per-size train overrides (e.g. the exact-fill batch for sizes started after 22:30 Oct 19)
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


def result_tokens(run_dir: Path):
    """Tokens reached by a finished stage (result.json progress), or None."""
    try:
        return int(json.load(open(run_dir / "result.json"))["progress"]["tokens"])
    except (OSError, KeyError, ValueError):
        return None


def read_past_one_epoch(run_dir: Path) -> str | None:
    """Description if a finished stage read more input tokens than its training split holds (the sampler silently wraps into a 2nd epoch),
    else None. Guards the 2026-10-19 finding: discarded micro-batch rows made runs read 1.96x the tokens they trained on."""
    try:
        r = json.load(open(run_dir / "result.json"))
        seen = int(r["progress"]["input_tokens_seen"])
        manifest = r["training_descriptor"]["configuration"]["data"]["manifest"]
        avail = int(json.load(open(manifest))["tokens"]["train"])
    except (OSError, KeyError, ValueError):
        return None
    return f"read {seen} input tokens > {avail} in the training split ({seen / avail:.2f} epochs)" if seen > avail else None


def parent_dirs(out: Path) -> list[Path]:
    """parent/ plus continuation dirs parent_c1, parent_c2, ... (resume always writes a NEW run dir)."""
    conts = sorted(out.glob("parent_c*"), key=lambda p: int(p.name.split("_c")[1]))
    return ([out / "parent"] if (out / "parent").exists() else []) + conts


def furthest_checkpoint(out: Path, batch: int):
    """(reached, (tokens, checkpoint)): tokens reached by any finished parent stage (a pruned final/ still counts, via result.json), and the
    furthest checkpoint that can actually be resumed (a final/ or recovery-<steps> whose state.pt still exists; (0, None) if none)."""
    reached, resumable = 0, (0, None)
    for d in parent_dirs(out):
        t = result_tokens(d)
        if t is not None:
            reached = max(reached, t)
            if (d / "final" / "state.pt").exists() and t > resumable[0]:
                resumable = (t, d / "final")
        for rec in d.glob("recovery-*"):
            t = int(rec.name.split("-")[1]) * batch
            if (rec / "state.pt").exists() and t > resumable[0]:
                resumable = (t, rec)
    return max(reached, resumable[0]), resumable


def find_milestone(out: Path, threshold: int):
    for d in parent_dirs(out):
        if (d / f"milestone-{threshold:012d}").exists():
            return d / f"milestone-{threshold:012d}"
    return None


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
    last = b["branch"][b["budgets"][-1]]
    # Parent: train to the largest budget's branch point. A runtime cap or crash leaves a shorter final/ or a recovery state; continue from the
    # furthest one into parent_c<k> (milestones are found in whichever parent dir saved them). Stops if an attempt makes no progress.
    reached, (start, ckpt) = furthest_checkpoint(out, batch)
    while reached < last:
        if ckpt is None:
            if (out / "parent").exists():
                (out / "parent").rename(out / f"parent_failed{len(list(out.glob('parent_failed*')))}")
            run_dir, extra, start = out / "parent", [], 0
        else:
            run_dir, extra = out / f"parent_c{len(list(out.glob('parent_c*'))) + 1}", ["--resume", str(ckpt)]
        print(f"STAGE_START {b['tag']}/{run_dir.name} from {start} tokens", flush=True)
        rc = sh(cli + ["train", "--config", str(cfg_path), "--run-dir", str(run_dir), "--max-steps", str((last - start) // batch),
                       "--max-seconds", str(max_seconds), "--device", device] + extra, out / "parent.log")
        before = reached
        reached, (start, ckpt) = furthest_checkpoint(out, batch)
        wrapped = read_past_one_epoch(run_dir)
        if wrapped:
            print(f"STAGE_FAILED {b['tag']}/{run_dir.name}: {wrapped}", flush=True)
            return 1
        if reached <= before:
            print(f"STAGE_FAILED {b['tag']}/{run_dir.name} rc={rc} (no progress past {before} tokens)", flush=True)
            return rc or 1
    for d in b["budgets"]:
        name = f"{b['tag']}_tpp{round(b['tpps'][d], 3):g}"  # D is rounded to the batch, so 4.99997 -> tpp5
        cand = out / name
        ev = cand / "evaluation" / "evaluation.json"
        if ev.exists():
            continue
        steps = (d - b["branch"][d] + batch - 1) // batch
        milestone = find_milestone(out, 9 * d // 10)
        resume = milestone if milestone is not None else furthest_checkpoint(out, batch)[1][1]  # largest budget: the parent's endpoint
        cand.mkdir(exist_ok=True)
        if not (cand / "cooldown" / "final").exists():
            print(f"STAGE_START {name}/cooldown D={d}", flush=True)
            rc = sh(cli + ["cooldown", "--config", str(cfg_path), "--resume", str(resume), "--cooldown-steps", str(steps),
                           "--cooldown-end-tokens", str(d), "--run-dir", str(cand / "cooldown"), "--max-steps", str(steps),
                           "--max-seconds", str(max_seconds), "--device", device], cand / "cooldown.log")
            wrapped = read_past_one_epoch(cand / "cooldown")
            if wrapped:
                print(f"STAGE_FAILED {name}/cooldown: {wrapped}", flush=True)
                return 1
            if rc or result_tokens(cand / "cooldown") != d:
                print(f"STAGE_FAILED {name}/cooldown rc={rc} tokens={result_tokens(cand / 'cooldown')} expected={d}", flush=True)
                return rc or 1
        if milestone is not None:
            drop_state(milestone)
        print(f"STAGE_START {name}/evaluation", flush=True)
        rc = sh(cli + ["evaluate", "--config", str(cfg_path), "--checkpoint", str(cand / "cooldown" / "final"), "--out", str(cand / "evaluation"),
                       "--max-seconds", "3600", "--device", device], cand / "evaluation.log")
        if rc:
            print(f"STAGE_FAILED {name}/evaluation rc={rc}", flush=True)
            return rc
        drop_state(cand / "cooldown")
        print(f"RUN_END {name} D={d} tpp={b['tpps'][d]:.2f} ce={json.load(open(ev))['ce']:.4f}", flush=True)
    for d in parent_dirs(out):
        drop_state(d)
        for rec in d.glob("recovery-*"):
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
    ap.add_argument("--write-job", action="store_true", help="write configs/scaling_grid/jobs/<tag>[_tpp<r>].sh for runs/queue.txt and exit")
    a = ap.parse_args(argv)
    tpps = [float(x) for x in a.tpp.split(",")]
    if a.plan:
        for s in yaml.safe_load(open(Path(a.grid_dir) / "sizes.yaml")):
            print(plan_line(build(s, 1.0, tpps, Path(a.grid_dir), Path(a.runs_dir))))
        return 0
    b = build(a.size, a.lr_mult, tpps, Path(a.grid_dir), Path(a.runs_dir))
    if a.write_job:
        args = [a.size] + ([f"--lr-mult {a.lr_mult:g}"] if a.lr_mult != 1 else []) + ([f"--tpp {a.tpp}"] if a.tpp != "5,10,20,40" else [])
        name = b["tag"] + (f"_tpp{a.tpp.replace(',', '-')}" if a.tpp != "5,10,20,40" else "")
        job = GRID / "jobs" / f"{name}.sh"
        job.write_text("#!/bin/bash\n# Queue entry for scripts/run_queue_file.sh (runs/queue.txt); written by scaling_grid.py --write-job.\n"
                       f'cd "$(dirname "$0")/../../.." && exec /venv/main/bin/python scripts/scaling_grid.py {" ".join(args)}\n')
        job.chmod(0o755)
        print(job.relative_to(ROOT))
        return 0
    print(plan_line(b), flush=True)
    return 0 if a.dry_run else run(b, a.device, a.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
