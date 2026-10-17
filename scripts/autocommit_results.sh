#!/bin/bash
# Hourly: refresh results/ from runs/ (scripts/export_results.py), commit it if anything changed (only results/), then push main
# (user-approved 2026-10-17). A failed push is logged and retried next hour; it never stops the loop.
cd "$(dirname "$0")/.." || exit 1
source /venv/main/bin/activate
while true; do
  OMP_NUM_THREADS=2 nice python scripts/export_results.py
  git add results
  if ! git diff --cached --quiet -- results; then
    git commit -q -m "results: export run curves and evaluations ($(date +%Y-%m-%d\ %H:%M))" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_011vAqPjkJ9gDBuVTNGtwg27" -- results \
      && echo "committed $(date +%H:%M)"
  fi
  GIT_ASKPASS=/bin/false GIT_TERMINAL_PROMPT=0 git push -q origin main 2>&1 | tail -1 | sed "s/^/push failed $(date +%H:%M): /"
  sleep 3600
done
