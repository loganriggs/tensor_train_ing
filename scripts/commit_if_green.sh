#!/bin/bash
# Commit (and push) only if the fast suite passes (skipped for docs/results-only commits), gated on pytest's EXIT CODE (not on grepping its output: with -q the last line is the
# warnings-docs link, which broke a grep gate on 2026-10-19). Adds the attribution trailers.
#   scripts/commit_if_green.sh "message" path [path ...]
set -u
cd "$(dirname "$0")/.." || exit 1
MSG=$1; shift
LOG=$(mktemp)
# Docs/results-only commits cannot change test outcomes: skip the ~2 min suite when every path is *.md, docs/, results/ or COORDINATION.md.
code_paths=$(for p in "$@"; do case "$p" in *.md|docs/*|results|results/*) ;; *) echo "$p";; esac; done)
if [ -z "$code_paths" ]; then echo "docs/results only: suite skipped"; else
if ! OMP_NUM_THREADS=2 nice /venv/main/bin/python -m pytest -n 6 -m "not slow" -p no:warnings > "$LOG" 2>&1; then
  echo "NOT COMMITTED: fast suite failed"; grep -E "^FAILED|^ERROR" "$LOG" | head -10; exit 1
fi
tail -1 "$LOG"
fi
git add -- "$@" && git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
  -m "Claude-Session: https://claude.ai/code/session_011vAqPjkJ9gDBuVTNGtwg27" || exit 1
git push -q origin main 2>&1 | sed 's#//[^@]*@#//***@#' | tail -1
echo "committed and pushed: $(git log -1 --format=%h)"
