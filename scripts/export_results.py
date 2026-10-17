"""Export the git-tracked record of every run under runs/: loss curves, evaluations and small summaries, but no weights.

    python scripts/export_results.py            # runs/ -> results/ (idempotent; rewrites changed files only)

`runs/` is git-ignored (84 GB: states, source snapshots, full result dumps). This writes a compact mirror to `results/`:
- `<run>/curve.json` from each metrics.jsonl. Columns step/tokens/loss/lr/grad_norm, at most MAX_ROWS rows (evenly thinned, last row
  always kept), plus every stability log (`stability.stats.worst`) and QK-clip record.
- `<run>/evaluation.json` from each evaluation.json. CE, document-bootstrap CI, training descriptor and progress, and per-document
  [loss_sum, tokens] in validation order. Document ids are stored once per validation set in `results/_validation_docs/<sha>.json`,
  which is enough to redo any paired comparison (`tnlm compare-evaluations` semantics) without the run directory.
- Small summaries copied verbatim (< 1 MB): sweep.json journals, comparisons, CSV/SVG plots, spend ledgers, markdown notes.
Skipped: Codex/Claude audit scratch dirs' `inputs/` copies, source snapshots, `result.json` (multi-MB dumps), `*.pt`.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "results"
MAX_ROWS = 2000
SMALL = 1_000_000
COPY_SUFFIX = {".csv", ".svg", ".md"}
COPY_NAMES = {"sweep.json", "spend.json", "comparisons.json", "pairs.yaml", "summary.json"}


def skip(p: Path) -> bool:
    parts = set(p.relative_to(RUNS).parts)
    return bool(parts & {"inputs", "source_snapshots"}) or any(s.endswith(".sources") for s in parts)


def write(dst: Path, obj) -> bool:
    text = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    if dst.exists() and dst.read_text() == text:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return True


def curve(src: Path) -> dict:
    cols = ("step", "tokens", "loss", "lr", "grad_norm")
    rows, stab, clips = [], [], []
    for line in open(src):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a run still writing its last line
            continue
        if "loss" in r and "step" in r:
            rows.append([r.get(c) for c in cols])
        st = (r.get("stability") or {}).get("stats")
        if st:
            stab.append({"step": r.get("step"), "worst": st.get("worst")})
        if r.get("qk_clip"):
            q = r["qk_clip"]
            clips.append({"step": r.get("step"), "clipped_heads": q.get("clipped_heads"), "max_before": q.get("max_before")})
    if len(rows) > MAX_ROWS:
        stride = len(rows) / MAX_ROWS
        rows = [rows[int(i * stride)] for i in range(MAX_ROWS - 1)] + [rows[-1]]
    return {"columns": list(cols), "rows": rows, "stability": stab, "qk_clip": clips, "source": str(src.relative_to(ROOT))}


def evaluation(src: Path, docs_dir: Path) -> dict:
    d = json.load(open(src))
    per = d.get("per_document") or []
    sha = d.get("validation_manifest_sha256", "unknown")
    write(docs_dir / f"{sha}.json", [x["document_id"] for x in per])
    keep = ("kind", "split", "context", "prediction_tokens", "ce", "uncertainty", "training_descriptor", "operator_audit",
            "validation_manifest_sha256", "training_manifest_sha256", "model_sha256", "evaluation_seed")
    out = {k: d[k] for k in keep if k in d}
    prog = dict(d.get("checkpoint_progress") or {})
    prog.pop("optimizer_exposure", None)  # per-parameter shape lists; large and recomputable
    out["checkpoint_progress"] = prog
    out["checkpoint"] = str(Path(d["checkpoint"]).relative_to(ROOT)) if str(d.get("checkpoint", "")).startswith(str(ROOT)) else d.get("checkpoint")
    out["per_document"] = [[x["loss_sum"], x["tokens"]] for x in per]
    out["source"] = str(src.relative_to(ROOT))
    return out


def main() -> int:
    changed = 0
    for src in sorted(RUNS.rglob("*")):
        if not src.is_file() or skip(src):
            continue
        rel = src.relative_to(RUNS)
        if src.name == "metrics.jsonl":
            changed += write(OUT / rel.parent / "curve.json", curve(src))
        elif src.name == "evaluation.json":
            changed += write(OUT / rel.parent / "evaluation.json", evaluation(src, OUT / "_validation_docs"))
        elif (src.suffix in COPY_SUFFIX or src.name in COPY_NAMES) and src.stat().st_size < SMALL:
            dst = OUT / rel
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                changed += 1
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"results/: {changed} files written, {size / 1e6:.1f} MB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
