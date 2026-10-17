# Bounded training timelines

`profile --trace-steps N` exports Chrome-format timelines for the first N complete post-warmup optimizer updates. N defaults to0 (disabled) and is limited to100. Capture works on CPU; a CUDA run includes both CPU and CUDA activities. `--kernels` can additionally retain the existing CUDA event ranking over every post-warmup update.

Example for an independently scheduled idle GPU window:

```bash
python -m tnlm.cli profile --config candidate.yaml --run-dir runs/profile-trace \
  --device cuda --warmup-steps 4 --steps 10 --trace-steps 3 --kernels \
  --max-seconds 300
```

Use a fresh output directory and a configuration with an explicit execution/budget policy, as for any profile run. Dry runs create no trace directory or profiler capture. CPU-only validation uses `--device cpu` without `--kernels`.

Files appear under the run's `traces/` directory. `profile.json` and the training result list trace paths, byte counts, SHA256 digests and zero-based local update indices. Each trace is exported to a temporary file and published atomically without replacing an existing file. An interrupted update is discarded; the profiler is closed without publishing it as complete. The count bound limits exported updates, not bytes per trace.

Named ranges distinguish `tnlm.sampler`, `tnlm.prepare_batch`, `tnlm.transfer`, `tnlm.forward`, `tnlm.validate_loss`, `tnlm.backward`, `tnlm.loss_accounting`, `tnlm.optimizer`, `tnlm.spectral_step` and CUDA `tnlm.synchronize`. All microbatches in the captured update are included. Torch operations outside named ranges remain visible in the timeline. Checkpointing and trace finalization/export are outside the capture; their training-process wall time remains billed.

Use traces to attribute work, and separate untraced runs to measure throughput: instrumentation changes timing and memory use. With trace-only mode, capture stops after the bounded trace count. With `--kernels`, CUDA ranking continues, but CPU timeline capture stops at the count bound. Post-warmup compilation/recompilation remains part of measured step time; a fixed warmup count does not prove that every input shape has compiled.

The outstanding trainer-versus-bare-loop comparison must use identical source batches, masks, optimizer work, dtype, accumulation and executed row counts. The earlier bare loop scored a different workload. This feature supplies the needed timeline capability; it does not establish the requested GPU throughput target.

Validation:81trainer/context/cosine/profiler integration tests passed in25.94seconds (onewarning). Final27capture/trace/CLI tests passed in1.81seconds, including real CPU Chrome export, named forward/backward ranges matching actual accumulation counts, exact traced/untraced model and optimizer-state equality, matching sampler state, warmup/count bounds, interrupted capture cleanup, no-overwrite publication and dry-run no-I/O. CUDA/CUPTI traces and GPU performance remain unverified.

Subsequent actual CUDA validation: [GPU efficiency audit](2026-10-16-codex-gpu-efficiency.md) records real trainer/bare captures and matched untraced throughput. Kineto GPU annotation ranges are now excluded from raw kernel totals. A real CUDA check agrees with Chrome raw-event count/duration;29CPU tests pass. Historical pre-fix kernel rankings can double-count annotations and should not be used.
