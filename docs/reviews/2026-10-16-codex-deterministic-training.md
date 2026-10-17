# Deterministic training policy integration

Claude's CUDA bisection (`runs/claude_cuda_determinism/README.md`) reports exact eager repeats and compiled differences localized to embedding gradients. His tiny compiled case becomes exact with PyTorch deterministic algorithms and an accepted cuBLAS workspace configuration. That result does not yet quantify throughput cost at101M parameters or certify every custom kernel or distributed backend.

Owned runtime, serial trainer and distributed trainer now support a strictly Boolean `cfg.train.deterministic` policy. Default remains false. The configuration-schema addition (`deterministic: bool = False` in Claude-owned `TrainConfig`) is requested in COORDINATION.md and is still pending at this writing. Until it lands, YAML rejects that new key; the runtime integration is not advertised as a finished CLI feature.

The wrapper establishes the policy before model setup/compilation. Enabled mode uses strict deterministic algorithms (not warn-only), disables cuDNN benchmarking and enables cuDNN deterministic behavior. It sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization unless an accepted setting (`:4096:8` or `:16:8`) is already present. If CUDA was already initialized with no accepted environment setting, it rejects and requires a fresh process. Merely changing an environment variable after handles have been cached is not sufficient evidence of determinism.

Algorithm and cuDNN flags are restored on successful return or failure. CPU-only calls restore the workspace environment too. Once a deterministic call initializes CUDA, its workspace setting remains in the process environment so subsequent sweep stages can reuse the same context without misrepresenting the cached configuration. Concurrent training calls within one process are unsupported because these PyTorch settings are process-wide.

Dry runs validate the Boolean field but do not change runtime settings. Distributed policy setup uses collective error propagation and restores settings on ranks that entered successfully when another rank rejects. Results and serial checkpoint runtime metadata record effective algorithm, warning, workspace and cuDNN settings. Distributed checkpoints record effective settings in progress. The configured policy is part of checkpoint configuration; ordinary resume rejects switching it. Legacy omission is treated as false, not retrospectively as a deterministic guarantee.

No CUDA experiments were launched for this integration; the earlier GPU window is closed. The next authorized GPU window must measure compiled deterministic throughput and check replay at the real workload before changing the default. CPU tests are bounded API/lifecycle/replay evidence, not CUDA/NCCL certification.

Integration gate:736CPU tests passed,4warnings,61.46seconds using pytest-xdist8, CUDA hidden and isolated temporary directories, excluding kernel/parallel files. This includes existing two-rank CPU replay and new strict-policy/lifecycle/provenance tests.

Final integrated gate after CLI/NCCL setup, reconciliation and canary integration: **772CPU tests passed,3warnings,64.02seconds**, xdist8, CUDA hidden, kernel/parallel files excluded. No new CUDA performance or replay claim follows from this gate.

## Subsequent authorized GPU evidence

The 10:11–10:21 window established exact 101M compiled BF16 fresh/restarted runtime replay at approximately13.7% lower throughput. See [full audit](2026-10-16-codex-deterministic-gpu.md). Schema support remains pending; the audit explicitly used a programmatic policy override. Earlier no-GPU statements above describe the initial integration stage.
