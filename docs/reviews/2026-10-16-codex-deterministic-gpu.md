# Deterministic CUDA trainer audit, 2026-10-16

The authorized 10:11–10:21 UTC GPU window completed and was released at 10:20:00 UTC (see the exact coordination marker). No files under `runs/confirmation/` were changed. All audit artifacts live in `runs/codex_deterministic_window/`.

## Results

The 101,069,312-parameter compiled BF16 trainer on real FineWeb data produced bitwise-equal model, optimizer, sampler, RNG, and progress states for two fresh 52-update deterministic runs and for a 26+26-update checkpoint resume. All batch hashes and per-step losses also match exactly. Checkpoint sizes and SHA-256 hashes were verified before CPU comparison. The frozen package snapshot verified after the experiment.

| Policy/run | Timed updates | Scored tokens/second | Seconds/update |
| --- | ---: | ---: | ---: |
| Off | 40 | 68,366.7 | 0.119824 |
| On, complete | 40 | 58,949.3 | 0.138967 |
| On, repeat | 40 | 59,026.6 | 0.138785 |

Each timing uses 12 warmup updates followed by 40 measured updates, totaling 327,680 measured valid prediction targets. Off/on batch hashes match. Enabled policy retains 86.23–86.34% of baseline throughput, a 13.66–13.77% reduction (approximately 15.8–16.0% longer updates). This is a small sequential timing sample, not a statistical confidence interval. Compilation after warmup would remain in step timers; setup, warmup, checkpoint saves and metric writes are outside the timed throughput and remain in runtime accounting.

## Scope and execution

The workload uses the confirmation base configuration copied from `configs/confirmation/base.yaml`: width512, context512, batch16, 8,192 scored targets/update, compiled BF16, pre-RMS, and the existing attention/loss implementation. Each stage runs in a fresh process against the same frozen package, config copies and captured stage driver. The LR schedule remains the original 50M-token horizon; this short audit lies within its first 100 warmup updates.

Claude-owned `TrainConfig` still lacked the requested deterministic field at snapshot time. Therefore the captured driver explicitly sets `cfg.train.deterministic` after normal configuration loading and calls the actual trainer/profile runtime. Effective strict algorithms, cuBLAS workspace `:4096:8`, and cuDNN flags are recorded in artifacts. This establishes runtime behavior; it does not certify YAML/CLI activation. That schema request remains open. Default false is retained because the measured performance cost is material; exact-replay experiments can opt in once schema support lands.

The original `on_a` stage stopped cleanly at43 updates (31 timed) under the trainer's conservative runtime guard: its slowest compilation-heavy step was reserved before admitting another step. It is preserved as partial evidence and excluded from the comparison table and exact-state comparisons. The initial harness correctly stopped on its incomplete endpoint. A separately recorded continuation launched fresh `on_complete`, `on_b`, `on_prefix`, and `on_resumed` outputs using the same frozen driver/package, a larger internal runtime allowance, and an external process-group deadline within the window. No existing stage was overwritten. The updated CPU comparison script selects these completed endpoints; its originally captured version remains in the immutable snapshot.

This verifies one single-GPU software/hardware/workload combination. It does not establish universal custom-kernel, distributed CUDA/NCCL, cross-platform, or long-run determinism, or equivalence between deterministic and nondeterministic trajectories.

## Evidence

- `comparison.json`: exact state comparisons and completed timing reports.
- `additional_checks.json`: snapshot verification, matched off/on batches and repeated/resumed losses.
- `stages.json`, `continuation_stages.json`, logs: commands, endpoints, timings and partial-stage history.
- `execution_mode.json`, per-run `audit-policy-invocation.json`: explicit programmatic-policy qualification.
- `code_snapshot/`: immutable source/config/driver capture.
- Per-run `final/`, `metrics.jsonl`, `result.json`, `profile.json` where applicable; separate audit `spend.json` records runtime charges.

No production source changed for this window. The prior integrated CPU gate remains 772 passed; only audit scripts, documentation and coordination were updated.
