#!/bin/bash
# Exit (printing the flagged rows) as soon as the most recently written metrics.jsonl under runs/ shows a stability flag from
# scripts/stability_trace.py. Meant to run as a background watcher so a live divergence surfaces within ~5 min instead of at RUN_END.
#   scripts/stability_watch.sh [SEEN_FILE]   (SEEN_FILE lists already-reported metrics paths, one per line)
cd "$(dirname "$0")/.." || exit 1
SEEN=${1:-/dev/null}
while true; do
  f=$(find runs -name metrics.jsonl -mmin -10 -path "*parent*" ! -path "*/inputs/*" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
  if [ -n "$f" ] && ! grep -qxF "$f" "$SEEN" 2>/dev/null; then
    # alert on abrupt jumps and absolute blow-ups only; steady growth (">10x first log" alone) is healthy for this model
    hits=$(/venv/main/bin/python scripts/stability_trace.py "$f" 2>/dev/null | grep -E -- "<--.*(jump|abs)")
    if [ -n "$hits" ]; then echo "STABILITY_FLAG $f"; echo "$hits" | head -5; exit 0; fi
  fi
  sleep 300
done
