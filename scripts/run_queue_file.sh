#!/bin/bash
# File-driven GPU queue: like scripts/run_queue.sh, but the entries live in a file that is re-read before each one, so experiments can be
# inserted / reordered / removed while the queue runs (edit the file; the running entry is unaffected).
#
#   scripts/run_queue_file.sh WAIT_PID QUEUE_FILE
#
# QUEUE_FILE: one run_queue.sh ENTRY per line (EXP[:MIN_TOK_S[:MAX_SECONDS[:KEEP]]] or path/to/script.sh); blank lines and # comments ignored.
# Each entry is removed from the file when it starts and appended to QUEUE_FILE.done with its exit code. Stops at the first failure (the
# failed entry stays recorded in .done; the rest of the file is untouched). Each entry runs via `scripts/run_queue.sh 0 ENTRY`.
set -u
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R" || exit 1
WAIT=$1; QF=$2
if [ "$WAIT" != 0 ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done; echo "pid $WAIT exited at $(date +%H:%M)"; fi
while :; do
  entry=$(grep -v '^\s*\(#\|$\)' "$QF" 2>/dev/null | head -1)
  [ -z "$entry" ] && break
  # remove the first occurrence of this exact line (later duplicates stay queued)
  awk -v e="$entry" 'BEGIN{d=0} { if (!d && $0 == e) { d=1; next } print }' "$QF" > "$QF.tmp" && mv "$QF.tmp" "$QF"
  "${RUN_QUEUE:-scripts/run_queue.sh}" 0 "$entry"; rc=$?
  echo "$entry rc=$rc $(date +%H:%M)" >> "$QF.done"
  [ $rc -ne 0 ] && { echo "QUEUE_FILE_FAILED $entry"; exit $rc; }
done
echo "QUEUE_FILE_DONE $(date +%H:%M)"
