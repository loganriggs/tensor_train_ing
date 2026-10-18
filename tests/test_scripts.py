"""Smoke tests for the reusable tools in scripts/ (not part of the tnlm package, but relied on by every sweep)."""
import pytest
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


def test_export_results_keeps_losses_drops_weights(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_results as er
    runs, out = tmp_path / "runs", tmp_path / "results"
    run = runs / "exp" / "cand" / "parent"
    run.mkdir(parents=True)
    lines = [{"step": s, "tokens": 8 * s, "loss": 10.0 - s / 1000, "lr": 1e-3, "grad_norm": 1.0} for s in range(1, 5001)]
    lines[99]["stability"] = {"stats": {"worst": {"S1S2_max_abs": 12.0}, "per_layer": []}}
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n{\"step\": 50")  # torn last line
    (run / "state.pt").write_bytes(b"weights")
    ev = run.parent / "evaluation"; ev.mkdir()
    (ev / "evaluation.json").write_text(json.dumps({
        "ce": 2.5, "validation_manifest_sha256": "abc", "checkpoint": "/elsewhere/final",
        "checkpoint_progress": {"steps": 5000, "optimizer_exposure": {"big": 1}},
        "per_document": [{"document_id": "d0", "loss_sum": 3.0, "tokens": 1}, {"document_id": "d1", "loss_sum": 2.0, "tokens": 1}]}))
    (runs / "exp" / "cand.sources").mkdir(); (runs / "exp" / "cand.sources" / "notes.md").write_text("snapshot")
    monkeypatch.setattr(er, "ROOT", tmp_path); monkeypatch.setattr(er, "RUNS", runs); monkeypatch.setattr(er, "OUT", out)
    er.main()
    c = json.loads((out / "exp/cand/parent/curve.json").read_text())
    assert len(c["rows"]) == er.MAX_ROWS and c["rows"][-1][0] == 5000 and c["stability"][0]["step"] == 100
    e = json.loads((out / "exp/cand/evaluation/evaluation.json").read_text())
    assert e["per_document"] == [[3.0, 1], [2.0, 1]] and "optimizer_exposure" not in e["checkpoint_progress"]
    assert json.loads((out / "_validation_docs/abc.json").read_text()) == ["d0", "d1"]
    assert not list(out.rglob("*.pt")) and not (out / "exp/cand.sources").exists()


def test_run_queue_runs_in_order_and_stops_at_first_failure(tmp_path):
    ok, bad, never = (tmp_path / f"{n}.sh" for n in ("ok", "bad", "never"))
    for f, body in ((ok, "echo ran-ok"), (bad, "echo ran-bad; exit 3"), (never, "echo ran-never")):
        f.write_text(f"#!/bin/bash\n{body}\n")
    r = subprocess.run(["bash", str(ROOT / "scripts/run_queue.sh"), "0", str(ok), str(bad), str(never)], capture_output=True, text=True)
    assert r.returncode == 3
    assert r.stdout.index("ran-ok") < r.stdout.index("ran-bad") and "ran-never" not in r.stdout
    assert "QUEUE_FAILED" in r.stdout and "QUEUE_DONE" not in r.stdout


def test_make_pairs_resolves_absolute_and_refuses_missing(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_pairs as mp
    runs = tmp_path / "runs"
    for c in ("a", "b"):
        e = runs / "exp/v2" / c / c / "evaluation"; e.mkdir(parents=True); (e / "evaluation.json").write_text("{}")
    out = tmp_path / "pairs.yaml"
    mp.main([str(out), "x=exp/v2/a,exp/v2/b"], runs=runs)
    import yaml
    p = yaml.safe_load(out.read_text())[0]
    assert Path(p["left"]).is_absolute() and p["right"].endswith("exp/v2/b/b/evaluation/evaluation.json")
    with pytest.raises(SystemExit):
        mp.main([str(tmp_path / "bad.yaml"), "x=exp/v1/a,exp/v2/b"], runs=runs)
    assert not (tmp_path / "bad.yaml").exists()


def test_run_queue_passes_keep_list_and_count(tmp_path):
    fake = tmp_path / "fake.sh"
    fake.write_text('#!/bin/bash\necho "ARGS exp=$1 min=$2 keep=[$3] max=$4 dev=$5"\n'); fake.chmod(0o755)
    r = subprocess.run(["bash", str(ROOT / "scripts/run_queue.sh"), "0", "norm_screen:40000:21600:a_s1,b_s1"],
                       capture_output=True, text=True, env={**__import__("os").environ, "RUN_CANDIDATES": str(fake)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ARGS exp=norm_screen min=40000 keep=[a_s1 b_s1] max=21600 dev=cuda" in r.stdout


def test_poly_init_scan_reads_tokens_like_the_trainer():
    """Regression for 2026-10-18: reading int32 token files as uint16 interleaved token 0 ('!') and corrupted a diagnosis.
    Scripts must use the shared reader, not a hand-written dtype."""
    src = (ROOT / "scripts/poly_init_scan.py").read_text()
    assert "read_split_tokens" in src and "np.fromfile" not in src



def test_norm_cost_table_metrics(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import math
    import norm_cost_table as nct
    site = {"denominator": {"q01": 0.5, "q99": 2.0}, "radial_over_direct_random_delta": {"q50": 0.02}}
    n_class = {"operator_class": "N", "normalizer_sensitivity": {"sites": {"a": site, "b": site}},
               "fixed_denominator": {"ce_live": 4.0, "ce_fixed": None, "nonfinite_metrics": ["ce_fixed"]},
               "ablation": {"components": {"L0.mlp": {"norm_mediated(live-clamped)": {"mean": -0.3}}}}}
    p_class = {"operator_class": "P", "normalizer_sensitivity": {"sites": {}}, "fixed_denominator": {}, "ablation": {"components": {}}}
    (tmp_path / "n.json").write_text(json.dumps(n_class)); (tmp_path / "p.json").write_text(json.dumps(p_class))
    n, p = nct.row(str(tmp_path / "n.json")), nct.row(str(tmp_path / "p.json"))
    assert n["sites"] == 2 and n["scalars"] == 2 and n["spread"] == 4.0 and n["radial"] == 0.02 and math.isinf(n["frozen_gap"]) and n["ablation"] == 0.3
    assert p["sites"] == 0 and p["frozen_gap"] == 0.0 and p["ablation"] is None


def test_find_orphan_states_lists_only_evaluated_old_states(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import os
    import time as _t
    import find_orphan_states as fos
    runs = tmp_path / "runs"
    done = runs / "exp/a/a/cooldown/final"; done.mkdir(parents=True); (done / "state.pt").write_bytes(b"x")
    (runs / "exp/a/a/evaluation").mkdir(); (runs / "exp/a/a/evaluation/evaluation.json").write_text("{}")
    live = runs / "exp/b/b/parent/final"; live.mkdir(parents=True); (live / "state.pt").write_bytes(b"x")  # no evaluation yet
    old = _t.time() - 3600
    for p in (done / "state.pt", live / "state.pt"):
        os.utime(p, (old, old))
    found = list(fos.orphans(runs, 30))
    assert found == [done / "state.pt"]
    assert list(fos.orphans(runs, 30, keep=["exp/a"])) == []
