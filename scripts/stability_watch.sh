#!/bin/bash
# Exit (printing the flagged rows) as soon as the most recently written metrics.jsonl under runs/ shows a stability flag from
# scripts/stability_trace.py, or its recent throughput falls below 80% of its steady state (scripts/throughput.py --drop),
# or free disk falls below DISK_FLOOR_GB (default 30). Meant to run as a background watcher so a live divergence surfaces within ~5 min instead of at RUN_END.
#   scripts/stability_watch.sh [SEEN_FILE]   (SEEN_FILE lists already-reported metrics paths, one per line)
cd "$(dirname "$0")/.." || exit 1
SEEN=${1:-/dev/null}
while true; do
  f=$(find runs -name metrics.jsonl -mmin -10 -path "*parent*" ! -path "*/inputs/*" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
  if [ -n "$f" ] && ! grep -qxF "$f" "$SEEN" 2>/dev/null; then
    # alert on abrupt jumps and absolute blow-ups only; steady growth (">10x first log" alone) is healthy for this model
    hits=$(/venv/main/bin/python scripts/stability_trace.py "$f" 2>/dev/null | grep -E -- "<--.*(jump|abs)")
    if [ -n "$hits" ]; then echo "STABILITY_FLAG $f"; echo "$hits" | head -5; exit 0; fi
    # throughput regression (e.g. the 2026-10-19 spectral 9% slowdown): recent rate < 80% of the run's steady state
    ratio=$(/venv/main/bin/python scripts/throughput.py --drop "$f" 2>/dev/null)
    if [ -n "$ratio" ] && [ "$ratio" != n/a ] && awk "BEGIN{exit !($ratio < 0.8)}"; then echo "THROUGHPUT_DROP $f recent/steady=$ratio"; exit 0; fi
  fi
  # free-disk floor (2026-10-20: /tmp test dirs took free space 51 -> 38 GB in an hour, unnoticed until a manual check)
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ -n "$free_gb" ] && [ "$free_gb" -lt "${DISK_FLOOR_GB:-30}" ]; then echo "DISK_LOW ${free_gb}G free (< ${DISK_FLOOR_GB:-30}G)"; df -h / | tail -1; exit 0; fi
  sleep 300
done
