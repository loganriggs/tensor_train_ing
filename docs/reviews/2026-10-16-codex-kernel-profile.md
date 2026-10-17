# CUDA event ranking in the training profiler

`tnlm profile --config CONFIG --run-dir NEW_RUN --steps 10 --warmup-steps 4 --max-seconds 300 --device cuda --kernels`

The optional flag adds `kernel_profile` to `profile.json` and the underlying
training `result.json`. Entries are sorted by total CUDA duration and include
event name, call count, microseconds per call, milliseconds per completed
post-warmup update, and fraction of summed device duration. The report includes
device memory operations as well as compute kernels; names are preserved without
heuristic attribution to model branches.

Capture covers each complete training update, including gradient accumulation,
normalization bookkeeping, clipping and optimizer work. Warmup updates, model
initialization and checkpoint writes are outside capture. The same training
trajectory, target quota, checkpoints and spend ledger are used. Profiler setup,
collection and aggregation remain inside billed training time. Profiling adds
overhead, so the accompanying throughput is an instrumented measurement.

Only raw CUDA events are summed. CPU operator summaries can include the same
device events and are excluded to prevent double counting. Concurrent streams
can overlap, so total device duration is not elapsed wall time or GPU utilization.
An interrupted update releases its profiler without counting it as completed.
An early token/runtime limit can produce fewer captured updates than requested,
including an empty ranking when no post-warmup update completes.

`--kernels` requires a CUDA device. Dry runs retain the requested flag without
starting a profiler or requiring a live GPU. Ordinary CPU profiling is unchanged.
CPU tests cover event accounting and lifecycle using controlled profiler events;
they do not establish CUDA/CUPTI availability or actual GPU timings.

Validation: 10 focused profiler tests and the full 528-test CPU suite pass
(137.20 seconds, three warnings; CUDA kernel/parallel test files excluded).
