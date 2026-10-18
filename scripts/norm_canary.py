"""CPU end-to-end canary for normalization packages through the real `tnlm train` path.

    python scripts/norm_canary.py OUT_DIR [--norms pre_rms lagged taper] [--layers 12] [--d-model 256] [--heads 4]
                                  [--steps 600] [--lr 1e-3] [--threads 6] [--model-set qk_init_scale=0.1 ...]

Trains each package on the 5M-token pilot set (context 64, 512 tokens/update) from `configs/norm_screen/base.yaml`, with
the warmup the schedule implies (max(100, 1% of steps)). Token-denominated settings (lagged_half_life_tokens) are rescaled
automatically by 512 / base tokens-per-update so they keep their per-update meaning (override with --model-set). Then it prints a thinned loss / taper-gate / anchor trace and, for lagged models,
the calibrated mean_sq per site. Use before giving any new package GPU time. Promoted from one-off scratch scripts used three times
on 2026-10-18; findings in docs/scaling.md ("Polynomial normalization canaries").
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PILOT = str(ROOT / "data/fineweb_pilot_5m/manifest.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--norms", nargs="+", default=["pre_rms", "lagged", "taper"])
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--base", default=str(ROOT / "configs/norm_screen/base.yaml"))
    ap.add_argument("--model-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra model fields, e.g. qk_init_scale=0.1 residual_alpha=0.5 (values parsed as YAML)")
    ap.add_argument("--train-set", nargs="*", default=[], metavar="KEY=VALUE", help="extra train fields, e.g. taper_anchor_lambda=1.0")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(a.base))
    dk = a.d_model // a.heads
    for norm in a.norms:
        cfg = json.loads(json.dumps(base))
        cfg["project"].update(spend_ledger=str(out / "spend.json"), gpu_hourly_rate_usd=0.0)
        cfg["model"].update(layers=a.layers, d_model=a.d_model, heads=a.heads, d_key=dk, d_value=dk, d_mlp=3 * a.d_model,
                            width_scaling_base=None, normalization=norm,
                            final_normalization=norm if norm in ("lagged", "taper") else "rms")
        # Token-denominated settings must keep their *per-update* meaning when the batch shrinks to 512 tokens/update
        # (a 1e6-token half-life is ~122 updates at the screen's 8192 tokens/update but ~1950 here, i.e. effectively frozen).
        ratio = 512 / base["train"]["global_prediction_tokens_per_update"]
        cfg["model"]["lagged_half_life_tokens"] = base["model"].get("lagged_half_life_tokens", 1.0e6) * ratio
        cfg["model"].update({k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in a.model_set)})
        cfg["data"].update(manifest=PILOT, context=64, micro_batch_size=8)
        cfg["evaluation"].update(validation_manifest=PILOT)
        cfg["train"].update(global_prediction_tokens_per_update=512, rung_total_tokens=512 * a.steps, compile=False, lr=a.lr,
                            stability_every_steps=50, qk_clip_every_steps=10)
        cfg["train"].update({k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in a.train_set)})
        if norm == "spectral_rms":
            cfg["train"].update(qk_clip_tau=0.0, qk_clip_every_steps=0)
        p = out / f"{norm}.yaml"; yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
        r = subprocess.run([sys.executable, "-m", "tnlm.cli", "train", "--config", str(p), "--run-dir", str(out / norm),
                            "--max-steps", str(a.steps), "--max-seconds", "7200", "--device", "cpu"], capture_output=True,
                           text=True, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": str(a.threads)})
        mpath = out / norm / "metrics.jsonl"
        rows = [json.loads(l) for l in open(mpath)] if mpath.exists() else []
        L = [(x["step"], round(x["loss"], 3), x.get("taper_gate_used"), x.get("taper_anchor")) for x in rows if "loss" in x]
        print(f"{norm}: rc={r.returncode}" + (f"  error: {r.stderr.strip().splitlines()[-1]}" if r.returncode else ""))
        for row in L[::max(1, len(L) // 8)] + L[-1:]:
            print("   step/loss/gate/anchor", row)
        res = out / norm / "result.json"
        if norm == "lagged" and res.exists():
            sites = json.load(open(res))["progress"].get("calibration", {}).get("sites", [])
            print("   calibrated mean_sq:", ", ".join(f"{s['site']}={s['mean_sq']:.2e}" for s in sites[::4] + sites[-1:]))


if __name__ == "__main__":
    main()
