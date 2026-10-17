# Screening checkpoint retention

For screening runs where a crash can mean rerunning the segment:

```yaml
project:
  keep_recovery_checkpoints: 0
  save_initial_checkpoint: false
  milestone_tokens: []
```

Defaults remain two recovery checkpoints and an initial checkpoint. `keep_recovery_checkpoints` accepts nonnegative integers; zero disables periodic recovery writes entirely. One retains the newest completed recovery, writing its replacement successfully before pruning the old one. Existing ledger-pinned recovery checkpoints remain protected and can exceed this rolling count. `save_initial_checkpoint` is a strict Boolean. False skips the initial full-state save for both fresh and resumed segments; it makes no new promise that recreating initialization from a seed is bitwise identical across machines or versions.

Final checkpoints, explicit milestones, and the mandatory 90% cooldown branch are unaffected. With no initial or recovery checkpoint, a crash before any milestone/final leaves no checkpoint in that segment. Failed replacement saves retain the preceding completed recovery. The policy is recorded in the full checkpoint config and in `checkpoint_storage_plan.json`; resume can change project execution settings under the existing validator. Serial and distributed trainers apply the same policy.

The storage planner removes disabled initial/recovery copies, includes replacement staging when recovery retention is positive, and preserves branch/final accounting. Sweep pruning forecasts now retain one final per segment when initial saving is disabled, rather than always assuming an initial+final pair. Sweeps still admit new work against actual available disk; they do not credit hypothetical deletion of existing files.

## Width-2048 planning check

Meta-only projection of `configs/width2048/base.yaml`, 50,003,968 tokens, empty ordinary milestones, parent plus cooldown (decimal GB):

| Recoveries retained | Save initial | Parent + cooldown projected states | Additional GB |
| ---: | :---: | ---: | ---: |
| 2 (default) | yes | 6 + 5 | 146.4 |
| 1 | no | 4 + 3 | 93.2 |
| 0 | no | 2 + 1 | 39.9 |

Thus zero recovery plus no initial fits the reported approximately91GB free; one recovery does not fit that figure. These remain conservative advisory projections, not a capacity reservation or a GPU-memory measurement. Normal filesystem admission still runs before launch. After successful evaluation and existing `--prune-completed-intermediates`, parent and cooldown finals retain about26.6GB per candidate. Multiple candidates still accumulate those finals; this policy alone does not make an arbitrary full grid fit on disk. Do not bypass admission.

Use a fresh source/config snapshot for subsequent launches. Claude owns the width2048 config edits and scheduling; the running joint sweep and its frozen sources are unchanged. No GPU work was required for implementation or validation.

Final validation: **846 CPU tests passed**, full suite with xdist8 and CUDA hidden, excluding kernel/parallel files. Includes serial keep0/1 and replacement-failure safety, two-rank CPU replay with0/1 and no initial, and a completed screening sweep through cooldown/evaluation.
