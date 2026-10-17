"""Smoke tests for the reusable tools in scripts/ (not part of the tnlm package, but relied on by every sweep)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_print_pairs_table(tmp_path):
    rec = {"pairs": [{"name": "a_vs_b", "left": "l", "right": "r",
                      "result": {"delta_ce_right_minus_left": -0.0123, "documents": 7,
                                 "uncertainty": {"delta_ce_95_ci": [-0.02, -0.004]}}}]}
    f = tmp_path / "comparisons.json"
    f.write_text(json.dumps(rec))
    out = subprocess.run([sys.executable, str(ROOT / "scripts/print_pairs.py"), str(f), "--markdown"],
                         capture_output=True, text=True, check=True).stdout
    assert "| a_vs_b | -0.012 | [-0.020, -0.004] | 7 |" in out


def test_check_experiment_configs_flags_default_retention(tmp_path, monkeypatch):
    import yaml
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_experiment_configs as cec
    base = yaml.safe_load(open(ROOT / "configs/joint50m/base.yaml"))
    good = tmp_path / "good"; bad = tmp_path / "bad"
    for d, retention in ((good, True), (bad, False)):
        d.mkdir()
        b = json.loads(json.dumps(base))
        if not retention:
            b["project"].pop("keep_recovery_checkpoints", None); b["project"].pop("save_initial_checkpoint", None)
        yaml.safe_dump(b, open(d / "base.yaml", "w"))
        yaml.safe_dump([{"name": "x", "overrides": {}}], open(d / "candidates.yaml", "w"))
    assert not [p for p in cec.check(good) if "retention" in p]
    assert any("retention" in p for p in cec.check(bad))


def test_stability_trace_flags_jump(tmp_path):
    def rec(step, s12):
        return {"step": step, "loss": 5.0, "stability": {"stats": {
            "worst": {"S1S2_max_abs": s12, "mlp_prod_max_abs": 10.0, "mlp_prod_excess_kurtosis": 1.0},
            "per_layer": [{"S1_rms": 1.0}]}}}
    f = tmp_path / "metrics.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [{"step": 1, "loss": 6.0}, rec(500, 100.0), rec(1000, 5000.0)]) + "\n")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/stability_trace.py"), str(f)],
                         capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if l.strip().startswith(("500", "1000"))]
    assert len(lines) == 2 and ">10x jump" in lines[1] and ">10x jump" not in lines[0]


def test_stability_trace_flags_steady_runaway(tmp_path):
    """A run that starts high and grows steadily (no single >10x step) must still be flagged (the 1B LR-1.35e-3 case)."""
    def rec(step, s12):
        return {"step": step, "loss": 5.0, "stability": {"stats": {
            "worst": {"S1S2_max_abs": s12, "mlp_prod_max_abs": 1.0, "mlp_prod_excess_kurtosis": 1.0},
            "per_layer": [{"S1_rms": 1.0}]}}}
    f = tmp_path / "metrics.jsonl"
    f.write_text("\n".join(json.dumps(rec(s, v)) for s, v in [(500, 2.0e5), (1000, 9.0e5), (1500, 3.0e6)]) + "\n")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/stability_trace.py"), str(f)],
                         capture_output=True, text=True, check=True).stdout
    rows = [l for l in out.splitlines() if l.strip()[:4] in ("500 ", "1000", "1500")]
    assert all("abs" in r for r in rows) and ">10x first log" in rows[-1] and ">10x jump" not in out
