#!/bin/bash
# Hourly: refresh results/ from runs/ (scripts/export_results.py) and commit it if anything changed. Commits only results/; never pushes.
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
  sleep 3600
done
