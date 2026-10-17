"""Preflight every experiment sweep config before queueing it on the GPU.

    python scripts/check_experiment_configs.py            # all configs/<exp>/base.yaml + candidates.yaml
    python scripts/check_experiment_configs.py joint50m   # one experiment
    python scripts/check_experiment_configs.py width2048 --keep-count 2   # include 2 kept cooled finals

For each experiment: loads base + candidates, reports the screening-retention settings, and plans one candidate with
`tnlm.sweep.plan_sweep`, comparing its staged bytes to current free disk. Exits non-zero on any problem. Added after two
sweeps were refused at disk admission because a config lacked screening retention (2026-10-16, 15:24 and 19:17).
"""
import shutil
import sys
from pathlib import Path

import yaml

from tnlm.config import Config
from tnlm.sweep import plan_sweep

ROOT = Path(__file__).resolve().parents[1]


def check(exp: Path, keep_count: int = 0) -> list[str]:
    problems = []
    cfg = Config.from_yaml(str(exp / "base.yaml"))
    project = cfg.raw.get("project", {})
    if project.get("keep_recovery_checkpoints", 2) != 0 or project.get("save_initial_checkpoint", True):
        problems.append("not using screening retention (keep_recovery_checkpoints: 0, save_initial_checkpoint: false)")
    if project.get("milestone_tokens") != []:
        problems.append("milestone_tokens is not [] (token-milestone full-state saves)")
    cands = yaml.safe_load(open(exp / "candidates.yaml"))
    names = [c["name"] for c in cands]
    if len(names) != len(set(names)):
        problems.append("duplicate candidate names")
    staged = max(plan_sweep(cfg, [c])["required_additional_bytes"] for c in cands)
    free = shutil.disk_usage(ROOT).free
    # kept cooled finals accumulate across the sweep: one full state each = 12 B/param (FP32 params + 2 Adam moments),
    # matching tnlm's checkpoint_storage_plan per_checkpoint_estimate_bytes
    from tnlm.config import ModelConfig
    from tnlm.shapes import count_params
    biggest = max(count_params(ModelConfig(**{**cfg.raw["model"], **c.get("overrides", {}).get("model", {})}))["total"]
                  for c in cands)
    per_state = 12 * biggest
    kept = keep_count * per_state
    # screening retention: parent stages branch+final (2 states), the cooldown fork adds one more
    peak = staged + per_state
    if peak + kept > free:
        problems.append(f"peak staging ~{peak / 1e9:.0f} GB (parent+cooldown) + {kept / 1e9:.0f} GB kept finals "
                        f"exceeds free disk {free / 1e9:.0f} GB")
    print(f"{exp.name:14s} candidates={len(cands):3d} parent_staged={staged / 1e9:6.1f} GB free={free / 1e9:5.0f} GB "
          f"{'OK' if not problems else 'PROBLEMS: ' + '; '.join(problems)}")
    return problems


def main(argv):
    keep = 0
    if "--keep-count" in argv:
        i = argv.index("--keep-count"); keep = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    exps = [ROOT / "configs" / a for a in argv] if argv else sorted(
        p for p in (ROOT / "configs").iterdir() if (p / "base.yaml").exists() and (p / "candidates.yaml").exists())
    bad = sum(bool(check(e, keep)) for e in exps)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
