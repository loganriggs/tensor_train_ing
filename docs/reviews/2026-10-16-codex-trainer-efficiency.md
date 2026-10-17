# Trainer workload and CPU packing improvements

The initial comparison was not equal work per optimizer update. In a read-only sample of the live confirmation metrics taken before Claude switched to its frozen snapshot, 198 of 200 updates had two batch hashes. Each scored 8,192 prediction targets. The bare loop in `scratch/profile_breakdown.py` performs one 16×512 forward/backward and labels every position, including document-boundary transitions. The trainer masks those transitions, finds slightly fewer than 8,192 targets in the first microbatch, and computes another full microbatch to fill the small shortfall. The bare loop also uses synthetic fixed document lengths; attention work can differ from real document packing.

A CPU replay of the first 200 training updates confirms 185 two-microbatch updates, 15 single-microbatch updates, 3,153,920 activation positions for 1,638,400 scored targets, and 1,511,285 discarded available targets. The second microbatch contributes only 3–23 scored targets (median 12). This is 1.925 activation positions per scored target. It plausibly explains a substantial portion of the reported 2.7× gap, but CPU counts alone cannot attribute GPU time.

Changes in the live development tree:

- PackedSampler builds within-document target masks using vectorized CPU tensors instead of assigning one scalar tensor element per token. The row packing, masks, lineage, epoch transitions and sampler cursor are unchanged.
- The serial trainer validates CPU tokens and computes its exact target quota before device transfer, eliminating device-to-host synchronization for those checks. It still validates model loss/counts and gradient finiteness.
- For deterministic models without any LaggedRMS site, rows with no scored targets are omitted from the forward/backward. Retained rows keep their entire sequence axis and causal prefixes. Sampler advancement, hashes and source exposure counters are unchanged. No data is carried into the next update and no target quota changes. Dropout and lagged-statistic collection disable this optimization because omitted rows could change RNG draws or EMA statistics.
- Metrics now record microbatch count, actual forward input positions and omitted zero-loss rows. `input_tokens_seen` continues to describe sampled source exposure; it is not relabeled as executed model positions.

The row omission is mathematically valid because current deterministic non-lagged operators do not mix batch rows: each retained loss and its parameter derivatives are unchanged, and the global valid-target divisor is unchanged. Floating-point reduction order may change; bitwise equality against older code is not asserted. Exact interruption/replay remains required within the same implementation. Keeping the sequence dimension unchanged limits the additional compiled shapes to batch-row counts instead of varying sequence lengths.

Measured CPU evidence: 400 alternating old/new 16×512 batches, exactly equal tensors, states and source mappings. Median construction time excluding the first ten batches was 20.163 ms before and 1.542 ms after, a 13.07× packing speedup. This is single-thread CPU timing, not whole-training throughput. Applying row omission to the replay reduces executed activation positions from 3,153,920 to 1,733,120 (45.05% fewer), or 1.0578 positions per scored target. Whole-step speedup is still unmeasured.

Evidence and reproducible scripts: `runs/codex_trainer_workload_audit/`. `baseline_data.py` preserves the pre-optimization sampler implementation; benchmark source hashes bind both versions. Only the train split is sampled. No GPU or final-test probes were used.

Next GPU validation, when Claude supplies an idle window: compare the old trainer, optimized trainer, and a bare loop using identical source batches, target masks, accumulation counts, clipping, optimizer, dtype, backend, and compilation settings. Warm all batch shapes, separate compile/setup from steady state, and use the existing `profile --kernels` instrumentation to count actual kernel work. Check loss/gradient/parameter trajectories with declared tolerances. Do not claim the ≥85%-of-bare target until the workloads match and a measured trace supports it. The current confirmation snapshot remains untouched and will retain its original training implementation.

Validation: 60 sampler tests, 17 independent objective/all-gradient/gating tests, and the final 651-test integrated CPU suite pass (177.79 seconds, three warnings; CUDA kernel/parallel files excluded). Current implementation and copied configs are available in `runs/source_snapshots/efficient_staging_v1`. No end-to-end GPU speedup is claimed.

## CPU compilation follow-up

A fullgraph Dynamo counting-backend audit on Torch 2.14.1+cu130 covers batch rows 16, 1, 2, 8, 9 and 15 at fixed sequence length 8. Chunk size 64 reproduces the B8→9 chunk-count boundary of the production T512/chunk4096 configuration. Four cases cross dense/chunked CE with activation checkpointing disabled/enabled. Document layouts and target masks vary between calls.

All 56 forward/backward comparisons match eager loss, valid-target count and every parameter gradient; maximum observed FP64 gradient error is zero. Alternating B16/B1 uses two captures. Dense CE settles at three graphs over the tested sizes; chunked CE uses six. Repeating those sizes adds no captures. Successful `fullgraph=True` execution rules out Dynamo graph breaks in these cases.

Evidence: `runs/codex_compile_shape_audit/reproduce.py` and `evidence.json`. The backend returns the captured GraphModule directly: it does not invoke Inductor, Triton or CUDA compilation and is not a performance benchmark. Different row sizes can introduce further specializations, particularly through the Python chunk loop; cache exhaustion across all possible shapes remains untested. GPU warmup must include both full and quota-filling batch shapes. No production code changed in this follow-up, and the confirmation directory was untouched.

## Recompilation-limit fix

Expanding the unbucketed audit to every row count 1–16 exposed a failure that the sparse shape sample missed: chunked CE reached Dynamo's default eight-recompilation limit at B8, raising `FailOnRecompileLimitHit` under fullgraph capture. Evidence and reproduction are `runs/codex_compile_shape_audit/all_rows.json` and `all_rows_reproduce.py`. The ordinary trainer does not request fullgraph capture; reaching its limit can instead fall back to eager execution.

The serial trainer now buckets row counts only for compiled deterministic, non-lagged forwards. For fixed original B, allowed sizes are the unique sorted values `{1, ceil(B/8), ceil(B/4), ceil(B/2), B}`. Every scored row is kept; extra rows are selected from original unscored rows, with source order and full sequence prefixes preserved. For B16 this yields 1/2/4/8/16, preserving the common B1 quota-filling path. Eager execution still trims exactly. Empty-target batches are skipped before the helper. Actual executed-position and omitted-row counters include bucket overhead.

This bounds input row shapes, not every possible compiler specialization: varying source B, other guards, CUDA kernels and Inductor can affect compilation. The existing source snapshot `efficient_staging_v1` predates this follow-up; existing snapshots and the confirmation directory remain unchanged. New tests exercise all row counts twice, changing target masks/document layouts, comparing full eager losses/counts/all parameter gradients against bucketed captured execution. They also check original-row identity and at-most-five sizes for B up to1024.

Follow-up validation: **127 tests passed, one warning, 50.36 seconds**, covering new bucket/capture tests plus trimming gates, trainer replay, context continuation, cosine, sweep and scaling-training integration. Independent review found no blocker. The earlier 651-test full-suite result predates this fix; this result is the focused integration gate. No GPU performance claim.

## Profile workload accounting

`profile.json` now totals microbatches, forward input positions and omitted unscored rows over exactly the same post-warmup update window as timed prediction targets. It reports positions per scored target and microbatches per update, returning null ratios when the timed window is empty. These quantities distinguish equal target quotas from equal executed work.

Timing language now states that compilation/recompilation after declared warmup remains measured. Step timers include batch preparation and forward/backward/optimizer work, but exclude checkpoint saves, metric writes and profiler finalization. Training initialization/warmup/saves remain in the training billing window; report postprocessing is outside that timer. A fixed warmup length does not certify that compilation is complete.

Validation: full integrated CPU gate before this reporting addition was685 passed/3warnings/188.30s; the reporting change then passed52 focused profiler/trainer tests in13.35s (onewarning), including empty/partial timed-window accounting. No CUDA performance measurement was made.

## Independent matched-work replay gate

The original `scratch/profile_breakdown.py` is absent from the live tree as of this follow-up. Its historical timing is not a reproducible current baseline. `tests/test_matched_training_replay.py` now supplies an independent CPU bare-loop replay from the trainer’s saved initial checkpoint. It independently applies the target quota and selects scored rows, retains complete sequence prefixes, sums microbatch losses, normalizes accumulated gradients by update target count, clips once and steps once. Learning rates are taken from recorded trainer groups; this checks replay, not an independent schedule formula.

The four cases cross AdamW/Muon with pre-RMS/spectral-RMS. Each uses updates of17,17,5targets, exercises multiple microbatches and omitted zero-loss rows, and compares original batch hashes, executed positions, loss and gradient norms. Final model tensors, optimizer tensors/metadata and sampler state match exactly. This establishes a small deterministic eager CPU reference trajectory; it does not establish compiled CUDA numerics, lagged-normalizer behavior or GPU throughput. A timed GPU baseline must additionally declare whether transfers and sampling are inside its timing scope and preserve the same executed workload.
