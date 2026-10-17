#!/bin/bash
# GPU queue after the width2048 sweep (pid 302296), reordered 12:25 after the seed-202 1B divergence:
# QK-clip 1B -> LR horizon -> critical batch -> exponent fit. Each preflights at launch and the queue stops at the first failure.
R=/workspace/tensor_train_ing; cd $R && source /venv/main/bin/activate
while kill -0 302296 2>/dev/null; do sleep 60; done
echo "width2048 queue exited at $(date +%H:%M)"
for exp in qkclip1b lr_horizon critical_batch exponent_fit; do
  echo "QUEUE_START $exp $(date +%H:%M)"
  if [ "$exp" = critical_batch ]; then
    bash runs/critical_batch/run.sh; rc=$?
  else
    PYTHONPATH=. python scripts/check_experiment_configs.py $exp || { echo "QUEUE_PREFLIGHT_FAILED $exp"; exit 1; }
    case $exp in lr_horizon) args='40000 "" 21600';; *) args='2500 "" 43200';; esac
    eval scripts/run_candidates_v2.sh $exp $args cuda; rc=$?
  fi
  echo "QUEUE_END $exp rc=$rc $(date +%H:%M)"
  [ $rc -ne 0 ] && { echo "QUEUE_FAILED $exp"; exit $rc; }
done
echo "QUEUE_DONE $(date +%H:%M)"
