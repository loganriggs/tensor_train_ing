#!/bin/bash
# CPU `tnlm analyze` on the kept cooled finals of a candidate sweep, each with its exact recorded config, then the norm cost table.
#
#   scripts/analyze_kept_finals.sh SWEEP_OUT_DIR TAG CAND [CAND ...]
#     e.g. scripts/analyze_kept_finals.sh runs/norm_screen/v1 ns pre_rms_lr0.00045_s101 branch_rms_lr0.00045_s101
#
# Configs come from `tnlm sweep-config` (checksum-verified recorded identity), never a hand merge of base + overrides. Outputs go to
# runs/analysis/TAG_CAND/; a candidate with an existing analysis.json is skipped. Failures are reported and left out of the table.
# Analysis output dirs are spend-ledger run IDs, so a retry after a failure needs a new TAG.
set -u
SWEEP=$1; TAG=$2; shift 2
cd "$(dirname "$0")/.." || exit 1
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
args=()
for c in "$@"; do
  ck=$SWEEP/$c/$c/cooldown/final; out=runs/analysis/${TAG}_$c; cfg=$out.config.yaml
  [ -d "$ck" ] || { echo "MISSING $ck"; continue; }
  if [ ! -f "$out/analysis.json" ]; then
    python -m tnlm.cli sweep-config --sweep "$SWEEP/$c" --candidate "$c" --out "$cfg" || { echo "CONFIG_FAILED $c"; continue; }
    OMP_NUM_THREADS=${THREADS:-12} nice python -m tnlm.cli analyze --config "$cfg" --checkpoint "$ck" --out "$out" --batches 2 \
      --rw-layer-window 4 --device cpu --max-seconds 3000 > "$out.log" 2>&1
  fi
  if [ -f "$out/analysis.json" ]; then echo "ANALYZED $c"; args+=("$c=$out/analysis.json"); else echo "ANALYZE_FAILED $c (see $out.log)"; fi
done
[ ${#args[@]} -gt 0 ] && python scripts/norm_cost_table.py "${args[@]}"
echo ANALYZE_KEPT_DONE
