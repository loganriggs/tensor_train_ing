# CUDA run-to-run reproducibility bisection (Claude, 2026-10-16 ~09:00)

`repro.py`: tiny polynomial model (2×256), 6 AdamW steps, BF16 autocast, each configuration run twice in-process.

| Configuration | max abs param diff between two identical runs | worst tensor |
| --- | ---: | --- |
| Triton attention + chunked CE, eager | 0 | – |
| reference attention + dense CE, eager | 0 | – |
| Triton + chunked CE, eager, deterministic algorithms | 0 | – |
| Triton + chunked CE, **torch.compile** | 1.2e-7 | `embed.weight` |
| Triton + chunked CE, torch.compile, `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | **0** | – |

The cause is Inductor's compiled embedding backward (atomic scatter-add), not the Triton attention kernels or chunked CE,
which reduce in a fixed order. Enabling deterministic algorithms restores bitwise replay under compile. The throughput cost
at 100M+ is not yet measured (needs an idle-GPU window).
