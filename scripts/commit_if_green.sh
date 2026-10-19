#!/bin/bash
# Commit (and push) only if the fast suite passes, gated on pytest's EXIT CODE (not on grepping its output: with -q the last line is the
# warnings-docs link, which broke a grep gate on 2026-10-19). Adds the attribution trailers.
#   scripts/commit_if_green.sh "message" path [path ...]
set -u
cd "$(dirname "$0")/.." || exit 1
MSG=$1; shift
LOG=$(mktemp)
if ! OMP_NUM_THREADS=2 nice /venv/main/bin/python -m pytest -n 6 -m "not slow" -q -p no:warnings > "$LOG" 2>&1; then
  echo "NOT COMMITTED: fast suite failed"; grep -E "^FAILED|^ERROR" "$LOG" | head -10; exit 1
fi
tail -1 "$LOG"
git add -- "$@" && git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_011vAqPjkJ9gDBuVTNGtwg27" || exit 1
git push -q origin main 2>&1 | sed 's#//[^@]*@#//***@#' | tail -1
echo "committed and pushed: $(git log -1 --format=%h)"
