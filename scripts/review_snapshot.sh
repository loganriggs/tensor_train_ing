#!/bin/bash
# One-command snapshot for the hourly code-health review: fast suite (exit-code gated), scratch policy, tree status, orphan states, data-epoch
# wraps, disk and /tmp, oversized modules, recent non-results commits. Prints a compact block to paste into docs/reviews/.
#   scripts/review_snapshot.sh
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=$(mktemp)
if OMP_NUM_THREADS=2 nice /venv/main/bin/python -m pytest -n 6 -m "not slow" -p no:warnings > "$LOG" 2>&1; then
  echo "suite: PASS ($(grep -Eo '[0-9]+ passed' "$LOG" | tail -1 || echo 'passed'))"
else
  echo "suite: FAIL"; grep -E "^FAILED|^ERROR" "$LOG" | head -5
fi
echo "scratch: $(ls scratch 2>/dev/null | wc -l) files; scratch imports in main repo: $(grep -rln '^from scratch\|^import scratch' tnlm tests scripts 2>/dev/null | wc -l)"
echo "tree: $(git status --porcelain | grep -vc ' results/') uncommitted non-results paths"
echo "orphan states: $(/venv/main/bin/python scripts/find_orphan_states.py | tail -1)"
echo "data epochs: $(/venv/main/bin/python scripts/check_data_epochs.py | grep -c WRAPPED) experiments with wrapped stages (3 known historical)"
echo "disk: $(df -h / | tail -1 | awk '{print $4}') free; /tmp $(du -xsh /tmp 2>/dev/null | cut -f1)"
echo "largest modules: $(wc -l tnlm/*.py | sort -rn | sed -n 2,4p | awk '{printf "%s %s; ", $2, $1}')"
echo "commits (last hour, excluding results exports):"; git log --since="1 hour ago" --format="  %h %s" | grep -v "results: export" | cut -c1-100
