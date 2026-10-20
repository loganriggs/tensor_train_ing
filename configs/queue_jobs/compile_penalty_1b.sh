#!/bin/bash
# Queue entry (runs/queue.txt): Codex's bounded serial eager-vs-compiled benchmark for the 24x2048 pilot model (red-team D1 item 2: the H100 cost
# estimate assumed compiled speed, but the distributed trainer can't compile). Runs Codex's isolated runtime worktree; bounded to 540 s.
# Command from /workspace/tensor_train_h100_runtime/docs/compile_penalty_benchmark.md.
cd /workspace/tensor_train_h100_runtime || exit 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  nice -n 10 /venv/main/bin/python scripts/benchmark_compile_penalty.py \
  --config configs/pilot_1b.yaml \
  --manifest /workspace/tensor_train_ing/data/fineweb_3b_heldout/manifest.json \
  --out /workspace/tensor_train_ing/runs/compile_penalty_1b_window \
  --context 2048 --micro-batch-size 1 --quota 2040 --rung-tokens 1000000 \
  --warmup 2 --steps 6 --timeout-seconds 540 --device cuda
