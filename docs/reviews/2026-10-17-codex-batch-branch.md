# Explicit batch-size-change checkpoint branches

Ordinary `train --resume` still rejects changes to global batch quota or LR. Use the new serial `branch` command for the critical-batch experiment:

```bash
python -m tnlm.cli branch \
  --config parent-config.yaml \
  --resume /path/to/parent/checkpoint \
  --batch-tokens 32768 --lr-scale 2 \
  --run-dir runs/critical-batch-32k \
  --max-steps 1000 --max-seconds 3600 --device cuda
```

For8192→32768 targets/update, `--lr-scale 2` gives sqrt(4) scaling. For8192→131072, use4. Scaling is explicitly supplied, not inferred or prescribed. Pass the **parent-compatible** config; do not pre-edit its batch or LR. It may increase `train.rung_total_tokens` using the existing upward-horizon rule. The branch derives a deep copy with the new batch and scaled primary/explicit auxiliary LR and writes `RUN/branch-config.yaml`. Parent config and checkpoint files are unchanged.

Later ordinary resume uses that child config, without repeating branch controls:

```bash
python -m tnlm.cli train \
  --config runs/critical-batch-32k/branch-config.yaml \
  --resume runs/critical-batch-32k/final \
  --run-dir runs/critical-batch-32k-continuation \
  --max-steps 1000 --max-seconds 3600 --device cuda
```

The cumulative token horizon must exceed the parent cursor. `max_steps` bounds new updates, and training stops at the configured cumulative token horizon, masking the final update to its exact remaining prediction-target quota. Neither command above is automatically launched by this change.

## State and intervention semantics

- Supports full serial checkpoints with `constant_with_separate_cooldown`, after their **recorded** warmup is complete. Preserves saved scheduler/warmup counters; no new warmup or reset. Cosine, cooldown parents, simultaneous context forks and distributed branches explicitly reject. Dry-run planning reports pending parent validation and does not read the checkpoint.
- Loads model, optimizer moments, optimizer step counters, sampler and RNG exactly. Multiplies each restored optimizer group's `lr` and `base_lr` by the supplied finite positive scale, including Muon's auxiliary AdamW groups. Missing legacy base rates use the restored post-warmup LR. Primary/explicit auxiliary config rates are scaled accordingly.
- Betas, momentum and weight decay remain unchanged. Consequently momentum/decay exposure per token changes with the batch/LR intervention; this command does not silently retune them. Historical optimizer-exposure totals remain and future updates accumulate their actual scheduled exposure.
- Leaves microbatch size, context, seed and sampler cursor untouched. The existing sampler consumes complete packed microbatches; exact update quotas can mask excess targets, recorded in `discarded_prediction_targets`. Different batch quotas can therefore lead to different target sets at equal scored-token totals. Exact cursor restoration is not a claim of identical data exposure across differently batched runs.
- Checkpoint `progress.batch_history` records the branch token/update boundary, old and new batch/LR segments, parent checkpoint identity, restored sampler state and optimizer group settings before/after scaling. Fresh training records its initial segment. Older checkpoints without history retain an explicit unknown prefix.
- Evaluation descriptors and comparisons include semantic batch/LR history without artifact-specific paths. Strict seed-pair comparisons require complete history and explicit `batch_history` allowance when histories differ, alongside LR/batch configuration differences. Legacy history remains unknown/partial rather than inferred equal; ordinary matched-target evaluation contrasts remain available.

Initial/final/milestone/recovery behavior follows the configured screening retention policy. With `save_initial_checkpoint: true`, the child's initial checkpoint records the restored state and scaled group rates before any new update. Disabling that save does not alter branch semantics.

## Validation

CPU regression tests cover real post-warmup AdamW and Muon checkpoints, initial weight/moment/sampler/RNG preservation, all-group LR scaling, parent-config immutability, unchanged ordinary-resume validation, exact interrupted replay and partial final quotas. Tests also cover CLI forwarding/dry runs and unsupported branch rejection, plus semantic history validation/comparison. No GPU jobs or existing queue/config/snapshot modifications were made.

Final integrated gate: **905 CPU tests passed**, xdist8 with CUDA hidden, excluding kernel/parallel test files. This includes existing two-rank CPU training/replay gates; distributed batch branches remain unsupported.
