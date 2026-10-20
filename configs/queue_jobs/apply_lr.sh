#!/bin/bash
# Queue step: set a grid size's LR in configs/scaling_grid/sizes.yaml from its LR checks before its main run (scripts/apply_grid_lr.py; fallback x0.36,
# the s38m ratio). Invoked through symlinks named apply_lr_<size>.sh: the size comes from the link name. s191m/s304m are checked at 5 tokens/param.
size=$(basename "$0" .sh); size=${size#apply_lr_}
case "$size" in s191m|s304m) tpp=5 ;; *) tpp=10 ;; esac
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
/venv/main/bin/python scripts/apply_grid_lr.py "$size" --tpp "$tpp" \
  && git commit -q configs/scaling_grid/sizes.yaml -m "Grid: $size LR set from its tpp$tpp LR checks (apply_grid_lr.py)" \
     -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_011vAqPjkJ9gDBuVTNGtwg27" \
  && git push -q origin main 2>&1 | sed 's#//[^@]*@#//***@#' | tail -1
exit 0
