#!/bin/bash
# Queue entry for scripts/run_queue_file.sh (runs/queue.txt).
cd "$(dirname "$0")/../../.." && exec /venv/main/bin/python scripts/scaling_grid.py s101m
