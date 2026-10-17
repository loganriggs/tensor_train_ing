# Checkpoint retention forecast and admission

Fresh serial and distributed training checks projected checkpoint storage before model setup or spend-ledger admission. Resumed trajectories obtain their precise cursor and cooldown horizon after loading, then check capacity before the first new save/update. Every save still checks available space independently. The run writes checkpoint_storage_plan.json and includes the plan/admission in result.json. validate-config and plan-search report a fresh single-process projection without allocating real model tensors.

## Configuring ordinary milestones

```yaml
project:
  milestone_tokens: []  # disable ordinary milestone saves for screening
```

An explicit list replaces all ordinary milestones, including the default100M cadence. Entries must be positive integers; duplicates are removed. Omitting the key retains1/2/4/8/16/32M and every100M. Initial/final saves, rolling recoveries and the90%branch checkpoint remain. Serial training merges coincident milestone/branch values; distributed training keeps its separately named branchpoint. Both trainers and the planner now use integer90% arithmetic and the same milestone policy.

## What is estimated

The meta-only planner counts parameter storage, mature AdamW or Muon+AdamW optimizer state, persistent buffers, distributed replication/shard padding allowances, serialization margin and fixed metadata allowance. Portable checkpoints deduplicate tied storage; DCP model entries are conservatively counted by name. Gradients and nonpersistent accumulators are absent from checkpoint payloads. Optional distributed weight export adds a separate artifact.

Reachable milestones are bounded by actual start tokens, token limit and max_steps. The rolling allowance is min(reachable_updates, keep_recovery_checkpoints+1), including a staging save before pruning. New concurrent pins can retain additional recoveries outside this window. Already-existing files are not added twice: the free-space query already reflects them.

For the actual303.6M width1024 pilot configuration,585 updates reach4,792,320 of4.8M requested targets. Defaults forecast six permanent checkpoints plus three rolling/staging slots:36,086,124,996 additional bytes, or about33.61GiB. This deliberately treats even initialization as a mature optimizer checkpoint and allows time-triggered recovery saves, so it exceeds the storage actually used when no rolling saves occur.

This is an advisory capacity estimate, not a reservation or guaranteed upper bound. Concurrent writers/pins, arbitrary large metadata, allocator/filesystem behavior and time-limit truncation remain outside the estimate. No artifacts are deleted. Whole-sweep capacity admission is separate from this per-trajectory forecast and remains work for the proposed sweep executor.

Validation covers custom/empty/default schedules, serial/distributed branch behavior, bounded resume/cooldown, huge horizons without constructing a range list, ties, dtype/Muon sizing, unchanged RNG, real checkpoint byte checks, and serial/collective rejection before initial publication. Fresh rejection leaves no run or ledger reservation.

Final integrated CPU regression: **439 passed, 3 warnings in116.40s (exit0)**. Includes owned slow distributed checks; excludes Claude kernel/parallel tests. Each invocation uses a unique explicit temporary base directory and failed-only retention to avoid concurrent shared pytest cleanup.
