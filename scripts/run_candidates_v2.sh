#!/bin/bash
# Run an experiment sweep one candidate at a time, in bounded disk space (v2 of run_candidates.sh).
#
#   scripts/run_candidates_v2.sh EXP MIN_TOK_S "KEEP_NAMES" [MAX_SECONDS] [DEVICE]
#
# Reads configs/EXP/base.yaml and configs/EXP/candidates.yaml (list of {name, overrides}). The candidate order is the
# file order. Each candidate runs as its own `tnlm sweep` with the tool's source+config snapshot (--snapshot-sources) and a
# throughput canary (--min-tok-s). After each candidate's evaluation, every state.pt is deleted except the cooled finals of
# candidates named in KEEP_NAMES. Metrics, evaluations and sweep journals are always kept.
#
# Env OUTV (default v1) selects the output subdirectory, e.g. after a killed attempt whose run IDs are already in the ledger.
# Differences from v1: no hand-made order.txt / one_*.yaml (derived here into runs/EXP/candidates/), completed
# candidates are skipped on re-run, and the driver STOPS at the first failure (admission refusal, canary failure,
# crash) instead of cycling through the remaining candidates. Prints RUN_START / RUN_END / <EXP>_DONE / <EXP>_FAILED.
set -u
EXP=$1; MIN_TOK_S=$2; KEEP=${3:-}; MAXS=${4:-21600}; DEVICE=${5:-cuda}
R=$(cd "$(dirname "$0")/.." && pwd); OUT=$R/runs/$EXP/${OUTV:-v1}; CAND=$R/runs/$EXP/candidates
mkdir -p "$OUT" "$CAND"
source /venv/main/bin/activate; cd "$R"
python3 - "$R/configs/$EXP/candidates.yaml" "$CAND" > "$R/runs/$EXP/order.txt" <<'EOF' || { echo "${EXP}_FAILED: bad candidates.yaml"; exit 2; }
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cands = yaml.safe_load(open(src))
names = [c["name"] for c in cands]
assert len(names) == len(set(names)), "duplicate candidate names"
for c in cands:
    open(f"{dst}/one_{c['name']}.yaml", "w").write(yaml.safe_dump([c], sort_keys=False))
print("\n".join(names))
EOF
while read -r n; do
  [ -z "$n" ] && continue
  if grep -q '"status": "complete"' "$OUT/$n/sweep.json" 2>/dev/null; then echo "SKIP $n (complete)"; continue; fi
  echo "RUN_START $n $(date +%H:%M) free=$(df -h / | tail -1 | awk '{print $4}')"
  python -m tnlm.cli sweep --config "configs/$EXP/base.yaml" --candidates "$CAND/one_$n.yaml" \
    --out "$OUT/$n" --max-seconds "$MAXS" --evaluation-max-seconds 1800 --device "$DEVICE" \
    --snapshot-sources --prune-completed-intermediates --canary-steps 60 --canary-warmup-steps 20 --min-tok-s "$MIN_TOK_S" \
    > "$OUT/$n.out.json" 2> "$OUT/$n.err.log"
  rc=$?
  ce=$(python3 -c "import json,glob; f=glob.glob('$OUT/$n/**/evaluation.json', recursive=True); print(round(json.load(open(f[0]))['ce'],4) if f else 'NA')" 2>/dev/null)
  find "$OUT/$n" -name state.pt 2>/dev/null | while read -r st; do
    case " $KEEP " in *" $n "*) [[ $st == */cooldown/final/* ]] && continue;; esac
    rm -f "$st"
  done
  echo "RUN_END $n rc=$rc ce=$ce free=$(df -h / | tail -1 | awk '{print $4}') $(date +%H:%M)"
  if [ $rc -ne 0 ]; then tail -4 "$OUT/$n.err.log"; echo "${EXP}_FAILED at $n"; exit $rc; fi
done < "$R/runs/$EXP/order.txt"
echo "${EXP}_DONE"
