# Distributed Muon recovery and mathematical checks

The distributed trainer now accepts native Muon on separate hidden matrices and AdamW on vocabulary/gain/scalar parameters. DCP receives the two child optimizers explicitly. Initial saves allocate zero momentum/moments without executing updates. Ordered logical parameter groups are bound in checkpoint layout; overlaps, duplicate membership, changed optimizer kind and changed group order are rejected. Same-world recovery retains both child states and scheduled learning rates.

## Mathematical boundary

Muon uses a matrix-level nonlinear transformation of momentum. Row sharding is a storage layout, not a new collection of independent optimizer matrices. A global norm and Gram-matrix products generally differ from the corresponding operations on separate local shards. Likewise, concatenating L and R changes the matrix transformation: computation fusion must preserve their separate optimizer identities.

The installed PyTorch 2.14 implementation casts its Newton–Schulz iterate to BF16. Its finite iteration is an approximate orthogonalization, not an exact polar decomposition. The configured match_rms_adamw rule scales the learning rate by 0.2 sqrt(max(rows, columns)), using logical matrix dimensions. See the [official Muon contract](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html). Tests exercise the installed implementation rather than assuming distributed support from this documentation.

## Evidence

Three updates with identical synthetic gradients, two CPU/gloo ranks, seed137 and learning rate0.01:

| Logical shape | DTensor vs serial max absolute weight difference | Incorrect shard-local vs serial |
|---|---:|---:|
| 8x4 | 0.0001932383 | 0.0034366846 |
| 4x8 | 0.0001324415 | 0.0030531883 |
| 7x5 | 0.0002580881 | 0.0062117577 |

Momentum is exactly equal in every case. The absolute weight tolerance4e-4, relative tolerance0, was fixed from earlier independent probes before these tests. All shard-local negative controls exceed it against both serial and DTensor updates. This bounds these fixtures; it is not a general error guarantee for larger matrices or longer training.

Separate two-rank tests cover pre-RMS, spectral-RMS and lagged normalization: no weight/RNG mutation during initial save, first true AdamW step counter1, exact next-batch/model/optimizer replay with a zero-target rank, and exact interrupted warmup/cooldown trajectories. CLI tests perform training, full evaluation export, held-out evaluation and reporting under both optimizers. Results retain training configuration and each step records all optimizer-group learning rates.

The [numeric artifact](2026-10-16-codex-distributed-muon.json) records measured errors, PyTorch version and source hashes. CUDA/NCCL, distributed compilation, Muon communication/peak-memory efficiency, longer-run quality and large-scale interpretation retention remain unverified. The optimizer change preserves the forward architecture and separate factors; it does not establish gauge-invariant training or better capability.

Combined CPU regression:396 passed,3 warnings in107.27s, no skips. Command: `CUDA_VISIBLE_DEVICES='' /venv/main/bin/python -m pytest -o addopts='' -q --ignore=tests/test_kernels.py --ignore=tests/test_parallel.py`. Claude GPU kernel/parallel tests were excluded; all owned slow distributed tests were included.
