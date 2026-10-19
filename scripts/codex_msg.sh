#!/bin/bash
# Send a message to Codex's tmux window and make sure it was SUBMITTED, not left sitting in the input box (2026-10-19: two messages sat
# unsent for 6 h because Enter arrived while Codex was busy).
#   scripts/codex_msg.sh "message text"      (target: $CODEX_PANE, default ssh_tmux:codex)
set -u
PANE=${CODEX_PANE:-ssh_tmux:codex}
MSG=$1
tmux send-keys -t "$PANE" -l "$MSG"
for attempt in 1 2 3 4 5; do
  tmux send-keys -t "$PANE" Enter
  sleep 3
  # Still unsent if the text's first 40 chars are on the input line ("› ..."; bottom 4 lines only: scrollback also shows "› " for sent messages).
  if ! tmux capture-pane -t "$PANE" -p | grep -v "^\s*$" | tail -4 | grep -F "› ${MSG:0:40}" >/dev/null; then echo "submitted (attempt $attempt)"; exit 0; fi
done
echo "WARNING: message still in Codex's input box after 5 Enters" >&2; exit 1
