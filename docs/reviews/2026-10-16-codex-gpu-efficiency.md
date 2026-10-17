# Idle-GPU trainer efficiency verification

GPU window: 2026-10-16 08:33–08:39:26 UTC, released before the08:48 deadline. GPU occupancy was checked before launch and after all audit processes exited. No writes under `runs/confirmation/`, no commits. Artifacts and scripts: `runs/codex_gpu_efficiency_20261016/`; `summary.json` binds small evidence files by SHA256. Checkpoint manifests independently bind state payloads.

## Matched throughput

The101,069,312-parameter pre-RMS model used compiled Triton polynomial attention, chunked CE4096, BF16 compute, B16×T512,8192scored targets/update and the frozen FineWeb training subset. Training used its actual AdamW configuration, not the fused AdamW of the historical bare-loop claim. The first12updates were excluded as warmup;40updates/327,680targets were timed.

| Measurement | Seconds/update | Scored targets/s |
| --- | ---: | ---: |
| Trainer | 0.12004 | 68,245 |
| Matched bare replay | 0.11580 | 70,745 |
| Independent trainer repeat | 0.11939 | 68,616 |

Trainer throughput was96.47% of bare throughput in the matched run, exceeding the requested85% threshold for this configuration and sample. Both use the same initial checkpoint, actual source batches, document masks, scored-target quotas, compiled row buckets, accumulation, clipping, optimizer and per-group learning rates. The bare replay prepares CPU batches outside timing but includes transfers and device synchronization. Thus the comparison gives the bare loop an explicit input-preparation advantage. It is not a universal efficiency or scaling result. Historical0.229-versus0.083seconds compared different workloads and optimizers; these new measurements are not a controlled old-versus-new implementation experiment.

The trainer executed78microbatches and347,136input positions during the40timed updates:1.95microbatches/update and1.059375input positions/scored target. Full-length prefixes are retained for each scored row. No targets were reordered to obtain the result.

## Actual CUDA traces and accounting correction

`traced_fixed/traces/` contains two complete trainer timelines (updates12/13); `bare_traces/` contains corresponding bare replay timelines. Their instrumented timings are not used in the throughput table. The trainer CPU ranges show roughly3.8–4.3ms sampling and0.95–1.04ms preparation per captured update. Loss validation/accounting ranges include waits for previously queued GPU work; their full duration is not removable Python overhead.

The first capture exposed a production profiling bug: Kineto emits GPU user-annotation events with CUDA device type. Counting all CUDA events added enclosing annotation ranges to kernel durations. `tnlm/profiling.py` now excludes `is_user_annotation`. The original `traced/profile.json` is preserved as faulty historical evidence; use `traced_fixed/profile.json` for the corrected ranking. The corrected five-update ranking contains10,605raw device events and no tnlm/Optimizer annotation rows. Dominant named attention kernels average15.19ms/update for backward KV,10.66ms for backward Q and5.96ms for forward. Summed device durations are not wall time.

Validation:29CPU profiler/trace tests pass, including synthetic CUDA annotation exclusions. A real tiny CUDA capture independently matches the Chrome raw-kernel count and duration:1kernel,14.08µs, with1GPU annotation excluded. The first tiny-check attempt incorrectly expected two nested GPU annotations; Kineto emitted one. That assertion was corrected and the rerun passed; the failed attempt and its ledger entry are retained.

## Numerical limits and follow-up

Original trainer versus bare replay: sampler equal, max final model-state absolute difference0.0015577, max step-loss difference0.000207995. Original versus independent trainer repeat: identical batches and sampler, max state difference0.0017164, max step-loss difference0.000168767. These establish measured drift, not a tolerance guarantee or explanation. CPU eager matched replay is exact; compiled BF16 CUDA replay is not established as exact. A request to investigate reproducibility is recorded for Claude; no kernel/model changes were made here.

The full364second reserved window costs approximately$0.04266 at the configured$0.4219/hour, including CPU/idle overhead. This is a whole-window estimate, not an invoice. The runtime ledger accounts for profile calls and tiny checks; standalone replay/setup overhead is covered only by the window estimate. No claim that the runtime ledger alone captures all experiment cost is made.
