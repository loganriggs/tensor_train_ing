#!/bin/bash
# Queue entry for scripts/run_queue_file.sh (runs/queue.txt); written by scaling_grid.py --write-job.
cd "$(dirname "$0")/../../.." && exec /venv/main/bin/python scripts/scaling_grid.py s67m --lr-mult 0.25 --tpp 10
