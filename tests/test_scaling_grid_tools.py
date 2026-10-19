"""Tests for the scaling-law grid tooling: scripts/scaling_grid.py (driver) and scripts/fit_scaling_law.py (fit)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_scaling_grid_end_to_end_cpu(tmp_path):
    """Parent with per-budget milestones, two cooled branches, evaluations, state pruning; then an idempotent re-run."""
    import yaml
    pilot = ROOT / "data/fineweb_pilot_5m/manifest.json"
    if not pilot.exists():
        pytest.skip("pilot data not prepared")
    base = yaml.safe_load(open(ROOT / "configs/scaling_grid/base.yaml"))
    base["model"].update(padded_vocab=50304)
    base["data"].update(context=32, micro_batch_size=4)
    base["evaluation"]["validation_manifest"] = str(pilot)
    base["train"].update(compile=False, bf16_compute=False, global_prediction_tokens_per_update=256, stability_every_steps=0,
                         qk_clip_every_steps=5)
    grid = tmp_path / "grid"; grid.mkdir()
    (grid / "base.yaml").write_text(yaml.safe_dump(base))
    (grid / "sizes.yaml").write_text(yaml.safe_dump({"tiny": {"model": {"d_model": 16, "heads": 1, "d_mlp": 32, "layers": 1},
                                                              "lr": 0.003, "data_manifest": str(pilot)}}))
    args = [sys.executable, str(ROOT / "scripts/scaling_grid.py"), "tiny", "--tpp", "0.003,0.006", "--device", "cpu",
            "--grid-dir", str(grid), "--runs-dir", str(tmp_path / "runs"), "--max-seconds", "600"]
    r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "runs/scaling_grid/tiny"
    ends = [l for l in r.stdout.splitlines() if l.startswith("RUN_END")]
    assert len(ends) == 2 and "GRID_DONE tiny" in r.stdout, r.stdout
    evs = sorted(out.glob("*/evaluation/evaluation.json"))
    assert len(evs) == 2
    actual = sorted(json.load(open(e))["training_descriptor"]["actual_prediction_tokens"] for e in evs)
    assert actual[1] == 2 * actual[0], actual  # each branch ends exactly at its own budget (the fit uses actual_prediction_tokens)
    cooled = sorted(json.load(open(e))["training_descriptor"]["cooldown_status"] for e in evs)
    assert cooled == ["cooled", "cooled"], cooled
    assert not list(out.rglob("state.pt")), "model states must be pruned"
    r2 = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0 and "STAGE_START" not in r2.stdout, r2.stdout


def test_scaling_grid_continues_a_time_capped_parent(tmp_path):
    """A runtime cap leaves a short parent final/; the driver must continue it (parent_c<k>) instead of branching from it."""
    import yaml
    pilot = ROOT / "data/fineweb_pilot_5m/manifest.json"
    if not pilot.exists():
        pytest.skip("pilot data not prepared")
    base = yaml.safe_load(open(ROOT / "configs/scaling_grid/base.yaml"))
    base["data"].update(context=32, micro_batch_size=4)
    base["evaluation"]["validation_manifest"] = str(pilot)
    base["project"].update(checkpoint_every_seconds=1)
    base["train"].update(compile=False, bf16_compute=False, global_prediction_tokens_per_update=256, stability_every_steps=0,
                         qk_clip_every_steps=5)
    grid = tmp_path / "grid"; grid.mkdir()
    (grid / "base.yaml").write_text(yaml.safe_dump(base))
    (grid / "sizes.yaml").write_text(yaml.safe_dump({"tiny": {"model": {"d_model": 16, "heads": 1, "d_mlp": 32, "layers": 1},
                                                              "lr": 0.003, "data_manifest": str(pilot)}}))
    common = [sys.executable, str(ROOT / "scripts/scaling_grid.py"), "tiny", "--tpp", "0.02", "--device", "cpu",
              "--grid-dir", str(grid), "--runs-dir", str(tmp_path / "runs")]
    r = subprocess.run(common + ["--max-seconds", "4"], capture_output=True, text=True, cwd=ROOT)
    out = tmp_path / "runs/scaling_grid/tiny"
    assert list(out.glob("parent_c*")), r.stdout + r.stderr  # the cap forced at least one continuation
    if r.returncode:  # cooldown/eval may also hit the tiny cap; finish with a normal cap
        r = subprocess.run(common + ["--max-seconds", "600"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "GRID_DONE tiny" in r.stdout, r.stdout + r.stderr
    ev = json.load(open(next(out.glob("*/evaluation/evaluation.json"))))
    assert ev["training_descriptor"]["actual_prediction_tokens"] == ev["training_descriptor"]["rung_total_tokens"]


def test_fit_scaling_law_recovers_chinchilla_parameters():
    sys.path.insert(0, str(ROOT / "scripts"))
    import fit_scaling_law as fsl
    true = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}  # Hoffmann et al. 2022
    pts = [(n, r * n, true["E"] + true["A"] / n ** true["alpha"] + true["B"] / (r * n) ** true["beta"])
           for n in (38e6, 67e6, 101e6, 189e6, 304e6) for r in (5, 10, 20, 40)]
    f = fsl.fit(pts)
    assert abs(f["alpha"] - 0.34) < 0.02 and abs(f["beta"] - 0.28) < 0.02 and abs(f["E"] - 1.69) < 0.05, f
    o, ref = fsl.optimal(f, 1e21), fsl.optimal(true, 1e21)
    assert abs(math.log(o["tokens_per_param"] / ref["tokens_per_param"])) < 0.15, (o, ref)  # within ~15%


def test_scaling_grid_detects_reading_past_one_epoch(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import scaling_grid as g
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"tokens": {"train": 1000}}))
    for seen, expect in ((900, None), (1960, "1.96 epochs")):
        d = tmp_path / f"run{seen}"; d.mkdir()
        (d / "result.json").write_text(json.dumps({"progress": {"input_tokens_seen": seen, "tokens": 1000},
                                                  "training_descriptor": {"configuration": {"data": {"manifest": str(manifest)}}}}))
        msg = g.read_past_one_epoch(d)
        assert (msg is None) if expect is None else (expect in msg), msg
