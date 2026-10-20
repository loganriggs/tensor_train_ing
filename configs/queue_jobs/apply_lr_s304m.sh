#!/bin/bash
# Queue step: set s304m's LR in configs/scaling_grid/sizes.yaml from its tpp5 LR checks (fallback x0.36, the s38m ratio) before its main run.
cd "$(dirname "$0")/../.." && /venv/main/bin/python scripts/apply_grid_lr.py s304m --tpp 5 && true \
  && git commit -q configs/scaling_grid/sizes.yaml -m "Grid: s304m LR set from its LR checks (apply_grid_lr.py)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
     -m "Claude-Session: https://claude.ai/code/session_011vAqPjkJ9gDBuVTNGtwg27" && git push -q origin main 2>&1 | sed 's#//[^@]*@#//***@#' | tail -1
exit 0
