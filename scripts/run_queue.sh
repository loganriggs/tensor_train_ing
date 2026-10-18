#!/bin/bash
# Run experiments back to back on the GPU, after an optional running job exits. Generalizes the hand-made runs/*/chain.sh files.
#
#   scripts/run_queue.sh WAIT_PID ENTRY [ENTRY ...]      (WAIT_PID 0 = start now)
#
# ENTRY is either
#   EXP[:MIN_TOK_S[:MAX_SECONDS[:KEEP]]] -> preflight (scripts/check_experiment_configs.py EXP --keep-count N), then
#                                     scripts/run_candidates_v2.sh EXP (defaults MIN_TOK_S=2500, MAX_SECONDS=43200, KEEP none;
#                                     KEEP = comma-separated candidate names whose cooled finals are kept for analysis), or
#   path/to/script.sh              -> run as-is (for experiments that are not candidate sweeps, e.g. runs/critical_batch/run.sh).
# Prints QUEUE_START / QUEUE_END / QUEUE_DONE, and stops at the first failure (QUEUE_FAILED / QUEUE_PREFLIGHT_FAILED).
# Never edit this file, or a running queue's entries, while a queue is executing it (bash reads scripts incrementally).
set -u
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R" || exit 1
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
WAIT=$1; shift
if [ "$WAIT" != 0 ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done; echo "pid $WAIT exited at $(date +%H:%M)"; fi
for entry in "$@"; do
  echo "QUEUE_START $entry $(date +%H:%M)"
  if [[ $entry == *.sh ]]; then
    bash "$entry"; rc=$?
  else
    IFS=: read -r exp mintok maxs keep <<< "$entry"
    keep=${keep//,/ }; nkeep=$(wc -w <<< "$keep")
    PYTHONPATH=. python scripts/check_experiment_configs.py "$exp" --keep-count "$nkeep" || { echo "QUEUE_PREFLIGHT_FAILED $exp"; exit 1; }
    "${RUN_CANDIDATES:-scripts/run_candidates_v2.sh}" "$exp" "${mintok:-2500}" "$keep" "${maxs:-43200}" cuda; rc=$?
  fi
  echo "QUEUE_END $entry rc=$rc $(date +%H:%M)"
  [ $rc -ne 0 ] && { echo "QUEUE_FAILED $entry"; exit $rc; }
done
echo "QUEUE_DONE $(date +%H:%M)"
