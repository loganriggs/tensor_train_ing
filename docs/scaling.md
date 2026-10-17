# Scaling status and measurements

Living document (Claude). All numbers were measured on one RTX 5000 Ada 32 GB, torch 2.14.1 / CUDA 13.0 /
Triton 3.8.0, BF16 autocast with FP32 params and optimizer state, full fwd+bwd+AdamW steps after warmup,
synthetic tokens with ~300-token documents. Re-measure on the target hardware before budgeting.

## Components
| Component | Where | Verified by |
| --- | --- | --- |
| Tiled polynomial attention fwd+bwd (3 variants, docs, padding, coef/b grads, document tile skipping) | `tnlm/kernels/poly_attn.py` | `tests/test_kernels.py` vs FP64 dense reference; FP32 rel < 2e-4, BF16 < 3e-2 |
| Chunked exact linear cross-entropy (no full logits) | `tnlm/kernels/fused_ce.py` | `test_math`, Codex's `test_scaling_loss` |
| Per-block activation checkpointing | `ModelConfig.activation_checkpointing` | exact grads in `test_math`; resume replay in `test_scaling_train` |
| Softmax control fast path (FlexAttention, causal+doc block mask) | `attention.softmax_attention_flex` | `test_kernels` vs dense SDPA |
| Muon (torch.optim.Muon) + AdamW split | `tnlm/optim.py` | `tests/test_optim.py` (exact resume) |
| FSDP2 sharding, loss-normalization contract, replicated lagged/spectral state | `tnlm/parallel.py` | `tests/test_parallel.py` (2-rank gloo == 1 process, <= 2e-13) |
| Shape presets 100M → 32B, memory planner | `tnlm/shapes.py` | `tests/test_shapes.py`; planner validated on 1.4B (below) |
| Width coordinate check, empirical width scaling | `tnlm/coordcheck.py`, `ModelConfig.width_scaling_base` | below |

## Throughput (compiled, Triton/Flex fast paths)
| Model | Context | Polynomial tok/s | Softmax control tok/s | Ratio | Poly peak GB |
| --- | --- | ---: | ---: | ---: | ---: |
| 100M (12×512) | 512 | 99.5k | 135k | 0.74 | 5.3 |
| 500M (24×1024) | 2048 | 23.6k | 32.0k | 0.74 | 17.7 |

The kernel alone is 14–38× faster than the dense FP32 reference. Document tile skipping gives a further
1.7× at T=2048 with 300-token docs.

**Where the remaining gap is** (100M, compiled, CUDA time per step; reproduce with `tnlm profile --kernels`):
the polynomial step takes 83 ms vs 62 ms for the control. GEMMs (26 ms vs 27 ms), compiled elementwise
(~29 ms vs 27 ms) and Adam (6 ms) are about equal. The difference is attention: poly fwd 3.8 + bwd_kv 10.4 +
bwd_q 7.5 = 21.7 ms vs ~3 ms for Flex softmax. The polynomial backward runs 11 tile matmuls across its two
kernels (S1, S2 and dP are recomputed in both), against about 5 for a FlashAttention-2-style softmax backward.
**Tried and rejected:** a single fused backward with atomic FP32 dQ (8 instead of 11 matmuls per tile) was
measured **slower**: 0.76× at T=512, 0.48× at T=2048, 0.13× at T=4096/hd128 (`deterministic` vs fused, interleaved
on a shared GPU). Atomic traffic grows as key-blocks × query-blocks × tile and dominates the saved recomputation. Removed.
The two-kernel path stays (it's also bitwise deterministic). Standalone microbenchmarks (about 1.07 ms/layer fwd+bwd) run faster
than in-model (about 1.8 ms/layer); sustained-load clocks on this power-limited card are the likely cause.
Always benchmark full steps.

## Memory planner check
`pilot_1b` (1.41B, head dim 128, checkpointing, chunked CE), 4096 tokens/microbatch on one GPU:
measured peak 22.7 GB vs planner 26.8 GB. The planner is about 15% conservative.

| Preset | Params | Min 80 GB GPUs (ckpt, 8K tok/GPU) | GB/GPU |
| --- | ---: | ---: | ---: |
| pilot_1b | 1.41B | 1 | 33 |
| pilot_3b | 2.81B | 1 | 58 |
| funded_7b | 6.91B | 2 | 73 |
| funded_32b | 34.4B | 16 | 69 |

Those are memory minimums. Throughput targets will need more GPUs.

## Width transfer (coordinate check)
Log-log slope vs width of the RMS change after 20 AdamW steps (lr 4.5e-4), 4 layers, head dim 64,
widths 256–2048, learnable repeat task:

| Parameterization | resid0 | resid3 | attn3 | mlp3 | logits | logits at init |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit conventional (default) | +0.57 | +0.54 | +0.48 | +0.49 | +0.90 | +0.50 |
| textbook μP-Adam, LR ∝ 1/d | −0.45 | −0.42 | −0.44 | −0.43 | −0.76 | −0.50 |
| `width_scaling_base` with exponents ½ | +0.07 | +0.12 | +0.09 | +0.09 | +0.07 | 0.00 |

Interpretation: under the default parameterization the per-step change grows about √width, so an LR tuned at
the 100M screen width is too large at 1B+. The ½ exponents flatten the coordinate check. Muon (match_rms_adamw)
shows similar +0.4–0.7 slopes.

## Width transfer (LR sweep on real data)
FineWeb pilot (manifest `38795155…`), 12 layers, head dim 64, AdamW, constant LR after 100 warmup steps,
585 updates × 8192 tokens ≈ 4.8M tokens per run, one seed, uncooled endpoints. Validation CE over
38,131 matched targets in 81 documents (`runs/lr_transfer/` (driver: `runs/lr_transfer/driver.py`)).

| LR | width 512 (101M) | width 1024 (304M) | width 1024 + `width_scaling_base: 512` |
| --- | ---: | ---: | ---: |
| 1.5e-4 | 6.217 | 6.005 | 6.074 |
| 4.5e-4 | **5.969** | **5.912** | **5.876** |
| 1.35e-3 | 6.017 | 6.195 | 6.030 |
| 4e-3 | 7.229 | 7.620 | 7.068 |

![Training loss curves for the width-transfer pilot, comparing learning rates at width 512, width 1024, and width 1024 with width scaling.](figures/scaling-width-training-loss.png)

Training cross-entropy from the saved `metrics.jsonl` files, plotted against cumulative scored prediction tokens. Faint lines are raw per-update losses; bold lines are trailing 25-update token-weighted means (shorter windows at the start). The dotted line marks the end of 100-update warmup. These are single-seed, uncooled **training** curves; the table above reports held-out validation CE. The scaled 4e-3 curve uses the completed `_retry` run. [Vector figure](figures/scaling-width-training-loss.svg).

Paired document-bootstrap deltas (`tnlm compare_evaluations`, 95% CI; nominal document uncertainty, not seeds):

| Comparison | ΔCE |
| --- | ---: |
| width 1024: width scaling vs standard @1.5e-4 | +0.069 [+0.061, +0.077] |
| … @4.5e-4 | −0.036 [−0.044, −0.026] |
| … @1.35e-3 | −0.165 [−0.181, −0.151] |
| … @4e-3 | −0.552 [−0.575, −0.523] |
| penalty for 3× LR (4.5e-4 → 1.35e-3), width 512 | +0.048 [+0.038, +0.058] |
| … width 1024 standard | +0.283 [+0.261, +0.310] |
| … width 1024 + width scaling | +0.154 [+0.140, +0.169] |

**Conclusions (scope: 2 widths, short warmup-heavy runs, 1 seed):**
- On this grid the best LR is 4.5e-4 at both widths under either parameterization, so a 3×-spaced grid can't
  resolve the √2 shift that width scaling applies.
- Under the standard parameterization, the high-LR side becomes much more costly with width (+0.05 → +0.28 for 3× LR).
  That's the danger the coordinate check predicted for 1B+ runs.
- `width_scaling_base` with exponents ½ roughly halves that width-induced sensitivity, beats the standard
  parameterization at every LR ≥ 4.5e-4, and gives the best result. It is the recommended setting for scaled configs
  (`configs/pilot_1b.yaml` … `funded_32b.yaml`) with base LR 4.5e-4 at width 512.
- Not yet established: transfer beyond 2×, depth and batch transfer, cooled endpoints, a second seed. Warmup is 17% of
  these rungs, above the brief's 10% flag. Treat the result as directional support, not a law.

## Depth (coordinate check)
Width 512, depths 2–16, 20 AdamW steps on the repeat task. Log-log slope vs depth, at init / per-step change:

| Setting | first-block resid | last-block resid | last attn | last mlp | logits |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | −0.08 / −0.09 | 0.00 / **+0.61** | −0.50 / +0.11 | −0.50 / +0.05 | 0.00 / **+0.47** |
| `depth_scaling_base` (global LR), exponent ½ | | / +0.19 | / −0.21 | / −0.35 | / +0.12 |
| exponent 1 | | / −0.29 | / −0.67 | / −0.82 | / −0.14 |

At init everything is depth-stable: branch outputs scale as L^−½ by design, so the residual stays O(1). Per step, each branch's
own change is about depth-flat, but the summed residual and logits change grows ~L^0.5–0.6. 12 → 64 layers (`funded_32b`)
means about 2.8× larger updates at the same LR. A single global LR factor trades this against early-layer learning:
exponent ½ flattens the output but makes first-block updates shrink as L^−0.6. That points to per-branch residual
scaling (CompleteP), not an LR knob. `depth_scaling_base` exists as an opt-in for experiments and is **off** in all configs.
Before any ≥ 32-layer run: a 12- vs 24-layer LR sweep on data and a CompleteP-style residual multiplier experiment.

**Per-branch residual scaling** (`depth_residual_base`, exponent 1: branch × Lb/L, D/O init at Lb). Coordinate check:
branch-internal updates become depth-flat (attn +0.11 → +0.02, mlp +0.05 → −0.03) and logits flat (+0.47 → −0.10).
That's CompleteP α=1 behaviour with no LR change.

**Depth LR sweep on data** (width 512, FineWeb pilot, same protocol as the width sweep; `runs/depth_transfer/`):

| LR | 12 layers | 24 layers default | 24 layers + `depth_residual_base: 12` |
| --- | ---: | ---: | ---: |
| 1.5e-4 | 6.217 | 6.145 | 6.187 |
| 4.5e-4 | 5.969 | 5.942 | **5.917** |
| 1.35e-3 | 6.017 | 6.116 | 6.008 |

![Training loss curves for the depth-transfer pilot, comparing learning rates for 12 layers, 24 layers, and 24 layers with residual scaling.](figures/scaling-depth-training-loss.png)

Same raw-loss and 25-update smoothing convention as the width curves; the 12-layer baseline is reused from that pilot. [Vector figure](figures/scaling-depth-training-loss.svg).

| Paired comparison (95% CI, documents) | ΔCE |
| --- | ---: |
| 24 L: residual scaling vs default @1.5e-4 | +0.042 [+0.037, +0.047] |
| … @4.5e-4 | −0.024 [−0.034, −0.015] |
| … @1.35e-3 | −0.108 [−0.124, −0.094] |
| penalty for 3× LR, 12 L / 24 L default / 24 L scaled | +0.048 / +0.175 / +0.090 |
| 24 L scaled best vs 12 L best | −0.051 [−0.060, −0.043] |

Same pattern as width: the default parameterization gets steadily more sensitive to the high-LR side as depth grows,
residual scaling halves that sensitivity and gives the best result, and the optimum on this 3× grid stays at 4.5e-4.
**Adopted** in the deep configs (`depth_residual_base: 12`) together with `width_scaling_base: 512`, base LR 4.5e-4.
Not tested: beyond-2× ratios (e.g. 32B: 10× width, 5.3× depth), cooled endpoints, seeds.

## Joint width × depth (24 × 1024, 2× both from the 12 × 512 base)
Same protocol (`runs/joint_transfer/`). "Scaled" = `width_scaling_base: 512` + `depth_residual_base: 12`.

| LR | 12×512 (101M) | 24×1024 default (504M) | 24×1024 scaled |
| --- | ---: | ---: | ---: |
| 1.5e-4 | 6.217 | 5.943 | 6.038 |
| 4.5e-4 | 5.969 | 5.926 | **5.830** |
| 1.35e-3 | 6.017 | 6.225 | 6.008 |

![Training loss curves for joint width and depth transfer, comparing the 101M baseline with standard and scaled 504M models.](figures/scaling-joint-training-loss.png)

Same training-loss convention, with the shared 101M baseline shown for comparison. [Vector figure](figures/scaling-joint-training-loss.svg). All three figures use completed pilot runs; the ongoing confirmation sweep is not included. [Metric sources and SHA-256 hashes](figures/scaling-training-loss-sources.json); regenerate with `python scripts/plot_scaling_losses.py` from the repository root.

| Paired comparison (95% CI) | ΔCE |
| --- | ---: |
| scaled vs default @1.5e-4 / @4.5e-4 / @1.35e-3 | +0.095 / **−0.097** [−0.110, −0.085] / −0.217 |
| penalty for 3× LR: base / default / scaled | +0.048 / +0.298 / +0.178 |
| 500M vs 100M at the base LR 4.5e-4: default | −0.043 [−0.053, −0.033] |
| … scaled | **−0.139** [−0.155, −0.127] |

The two rules compose. At the transferred base LR, the default parameterization recovers less than a third of the
improvement that the 5×-larger model gets with scaling, because its effective step is already past the optimum at this
size. It's the same failure the coordinate checks predicted, and it compounds with every doubling. The scaled
parameterization is the recommended default for all scaled configs. 50M-token cooled 2-seed confirmation: running
(`configs/confirmation/`).

## Confirmation at 50M tokens (cooled, independent 320M subset, 2 seeds)
`tnlm sweep` per candidate from a frozen code snapshot (tree 0478b67f), 6104 updates × 8192 tokens + 10% cooldown
branch, evaluated on the confirmation subset's validation split (1476 documents). Runs in `runs/confirmation/v3/`,
all 24 paired comparisons in `runs/confirmation/comparisons_all/` (spec `runs/confirmation/pairs.yaml`).

Validation CE (seed 101 / seed 202):

| LR | width 512 | width 1024 default | width 1024 + `width_scaling_base: 512` |
| --- | ---: | ---: | ---: |
| 4.5e-4 | 4.801 / 4.790 | 4.680 / 4.671 | **4.661 / 4.643** |
| 1.35e-3 | 4.814 / 4.796 | 4.882 / 4.822 | 4.743 / 4.738 |

Key paired effects (ΔCE; both seeds):

| Effect | seed 101 | seed 202 |
| --- | ---: | ---: |
| 1024 + scaling vs 1024 default @4.5e-4 | −0.019 | −0.028 |
| 1024 + scaling vs 1024 default @1.35e-3 | −0.139 | −0.084 |
| 1024 default vs 512 @1.35e-3 (the wider model is worse) | +0.068 | +0.027 |
| 3× LR penalty, width 512 | +0.013 | +0.006 |
| 3× LR penalty, width 1024 default | +0.202 | +0.152 |
| 3× LR penalty, width 1024 + scaling | +0.082 | +0.095 |
| seed 202 − seed 101, same config (range over 6 configs) | | −0.060 … −0.005 |

**What exactly was validated:** `optim.make_optimizer` scales hidden AdamW LR by r=(512/d)^0.5 **and divides hidden weight
decay by r**, so LR × WD stays constant. The tested rule is therefore "LR ∝ width^−½ *and* WD ∝ width^+½". Kosson et al.
(ICLR 2026) suggest exactly this combination as likely better than μP but did not test it (see `docs/literature/scaling_review.md`).
The width-2048 sweep adds a fixed-WD arm to separate the two.

**Uncertainty:** document-bootstrap CIs are about ±0.005 (full table below), but same-config seed differences reach
0.01–0.02, and **0.06 for default width 1024 at high LR**. Seed variance, not document sampling, is the binding error bar.
The high seed sensitivity of that cell is itself a symptom of the instability the parameterization removes.

**Conclusions (2 widths, 2 seeds, 50M cooled tokens, independent data), stronger than the pilot:**
- Every effect has the same sign under both seeds, and its magnitude exceeds the seed-to-seed spread.
- Under the default parameterization, an LR that is nearly harmless at width 512 makes the 2×-wider model *worse* than
  the narrow one, with both seeds.
- `width_scaling_base` cuts the width-induced high-LR penalty by ~40–60% and is better than the default at both LRs and both seeds.
- Not established: >2× width ratios, the depth/joint rules at this token count (pilot only), H100/multi-GPU numerics.

**Mechanism: attention score blow-up** (`analysis.stability_stats`, one 512-token validation row, cooled seed-101 finals):

| Checkpoint | max \|S1·S2\| | last-layer S1 RMS | MLP product max | MLP excess kurtosis |
| --- | ---: | ---: | ---: | ---: |
| 512, LR 1.35e-3 | 1,606 | 2.8 | 654 | 2,000 |
| **1024 default, LR 1.35e-3** | **408,965** | **39.7** | **7,009** | **24,701** |
| 1024 + width scaling, LR 1.35e-3 | 3,242 | 3.6 | 1,277 | 14,893 |
| 1024 default, LR 4.5e-4 | 1,956 | 2.6 | 291 | 489 |
| 1024 + width scaling, LR 4.5e-4 | 1,285 | 2.0 | 212 | 123 |

The configuration with the high-LR loss penalty and the largest seed variance has a degree-5 attention product ~125× larger than
its width-scaled twin, last-layer scores ~11× larger in RMS, and much heavier MLP tails. Under the default parameterization the
penalty comes from the polynomial scores growing, not from a mild optimization slowdown. **Guard for ≥1B runs:** log
`stability_stats` at checkpoints and treat a jump in S1·S2 max of more than ~10× as an instability signal. QK-clip is the
first remedy to try (review rec. 7), since it keeps the model exactly polynomial.

<details><summary>All 24 paired comparisons (document-bootstrap 95% CI)</summary>

| Comparison | ΔCE (right − left) | 95% CI | docs |
| --- | ---: | --- | ---: |
| s101:lr0.00045:1024_vs_512 | -0.121 | [-0.126, -0.116] | 1476 |
| s101:lr0.00045:1024ws_vs_512 | -0.140 | [-0.145, -0.133] | 1476 |
| s101:lr0.00045:1024ws_vs_1024 | -0.019 | [-0.022, -0.015] | 1476 |
| s101:lr0.00135:1024_vs_512 | +0.068 | [+0.048, +0.092] | 1476 |
| s101:lr0.00135:1024ws_vs_512 | -0.071 | [-0.075, -0.067] | 1476 |
| s101:lr0.00135:1024ws_vs_1024 | -0.139 | [-0.163, -0.118] | 1476 |
| s101:w512:lr3x_penalty | +0.013 | [+0.009, +0.018] | 1476 |
| s101:w1024:lr3x_penalty | +0.202 | [+0.182, +0.226] | 1476 |
| s101:w1024ws:lr3x_penalty | +0.082 | [+0.076, +0.086] | 1476 |
| s202:lr0.00045:1024_vs_512 | -0.119 | [-0.126, -0.111] | 1476 |
| s202:lr0.00045:1024ws_vs_512 | -0.147 | [-0.152, -0.142] | 1476 |
| s202:lr0.00045:1024ws_vs_1024 | -0.028 | [-0.033, -0.024] | 1476 |
| s202:lr0.00135:1024_vs_512 | +0.027 | [+0.024, +0.029] | 1476 |
| s202:lr0.00135:1024ws_vs_512 | -0.058 | [-0.061, -0.055] | 1476 |
| s202:lr0.00135:1024ws_vs_1024 | -0.084 | [-0.087, -0.082] | 1476 |
| s202:w512:lr3x_penalty | +0.006 | [+0.003, +0.009] | 1476 |
| s202:w1024:lr3x_penalty | +0.152 | [+0.144, +0.160] | 1476 |
| s202:w1024ws:lr3x_penalty | +0.095 | [+0.089, +0.102] | 1476 |
| seed202_vs_101:w512:lr0.00045 | -0.011 | [-0.015, -0.007] | 1476 |
| seed202_vs_101:w512:lr0.00135 | -0.018 | [-0.023, -0.013] | 1476 |
| seed202_vs_101:w1024:lr0.00045 | -0.009 | [-0.015, -0.004] | 1476 |
| seed202_vs_101:w1024:lr0.00135 | -0.060 | [-0.082, -0.039] | 1476 |
| seed202_vs_101:w1024ws:lr0.00045 | -0.018 | [-0.023, -0.014] | 1476 |
| seed202_vs_101:w1024ws:lr0.00135 | -0.005 | [-0.010, -0.000] | 1476 |

</details>

## Joint width × depth at 50M tokens (cooled, independent subset, 2 seeds)
24×1024 (~504M) default vs `width_scaling_base: 512` + `depth_residual_base: 12`, against the 12×512 confirmation baseline. Same
protocol as the width confirmation (`configs/joint50m/`, runs in `runs/joint50m/v1/`, comparisons in `runs/joint50m/comparisons/`).
Seed-202 candidates ran with screening retention and `stability_every_steps: 500`, which are execution-only settings (same math).

Validation CE (seed 101 / seed 202):

| LR | 12×512 (101M) | 12×1024 + width scaling (304M) | 24×1024 default (504M) | 24×1024 scaled (504M) |
| --- | ---: | ---: | ---: | ---: |
| 4.5e-4 | 4.801 / 4.790 | 4.661 / 4.643 | 4.675 / 4.662 | **4.620 / 4.615** |
| 1.35e-3 | 4.814 / 4.796 | 4.743 / 4.738 | 4.958 / 4.931 | 4.713 / 4.716 |

| Paired effect | seed 101 | seed 202 |
| --- | ---: | ---: |
| scaled vs default @4.5e-4 | −0.055 | −0.047 |
| scaled vs default @1.35e-3 | −0.246 | −0.215 |
| 500M default vs 100M baseline @1.35e-3 (the bigger model is *worse*) | +0.145 | +0.135 |
| 500M scaled vs 100M baseline @1.35e-3 | −0.101 | −0.080 |
| 3× LR penalty, default / scaled | +0.284 / +0.093 | +0.269 / +0.101 |

**Conclusions (2 seeds, 50M cooled tokens):** the pilot result holds at 10× the tokens. The two rules compose. With them, doubling depth
on top of doubling width is a clear gain (24×1024 scaled 4.620 vs 12×1024 scaled 4.661). With the default parameterization, depth adds almost nothing
at the base LR (4.675 vs 4.680 for 12×1024 default), and at a 3× LR the 5×-larger model is worse than the 100M baseline with both seeds.
Stability stats on the base-LR checkpoints show the same signature as the width study: the default has heavier MLP tails (kurtosis 1451 vs 271)
and larger score products (4.6k vs 3.4k) than its scaled twin (`runs/joint50m/stability_lr4.5e-4_s101.json`).

<details><summary>All 16 joint paired comparisons (document-bootstrap 95% CI)</summary>

| Comparison | ΔCE (right − left) | 95% CI | docs |
| --- | ---: | --- | ---: |
| s101:0.00045:scaled_vs_default | -0.055 | [-0.059, -0.052] | 1476 |
| s101:0.00045:default_vs_base512 | -0.126 | [-0.130, -0.122] | 1476 |
| s101:0.00045:scaled_vs_base512 | -0.181 | [-0.186, -0.176] | 1476 |
| s101:0.00135:scaled_vs_default | -0.246 | [-0.252, -0.239] | 1476 |
| s101:0.00135:default_vs_base512 | +0.145 | [+0.139, +0.150] | 1476 |
| s101:0.00135:scaled_vs_base512 | -0.101 | [-0.105, -0.097] | 1476 |
| s101:J24x1024:lr3x_penalty | +0.284 | [+0.275, +0.291] | 1476 |
| s101:J24x1024sc:lr3x_penalty | +0.093 | [+0.089, +0.097] | 1476 |
| s202:0.00045:scaled_vs_default | -0.047 | [-0.053, -0.043] | 1476 |
| s202:0.00045:default_vs_base512 | -0.127 | [-0.132, -0.120] | 1476 |
| s202:0.00045:scaled_vs_base512 | -0.174 | [-0.178, -0.171] | 1476 |
| s202:0.00135:scaled_vs_default | -0.215 | [-0.225, -0.201] | 1476 |
| s202:0.00135:default_vs_base512 | +0.135 | [+0.131, +0.140] | 1476 |
| s202:0.00135:scaled_vs_base512 | -0.080 | [-0.087, -0.067] | 1476 |
| s202:J24x1024:lr3x_penalty | +0.269 | [+0.261, +0.276] | 1476 |
| s202:J24x1024sc:lr3x_penalty | +0.101 | [+0.093, +0.114] | 1476 |

</details>

## Width 2048 (~1B) results (base LR: 2 seeds; high LR and fixed-WD arms: seed 101 / pending)
Same 50M-token cooled protocol on the confirmation subset (`configs/width2048/`, `runs/width2048/v2/`). Activation checkpointing,
screening retention, live stability logging every 500 steps (`scripts/stability_trace.py`).

| 12 layers, LR 4.5e-4, seed 101 | default | `width_scaling_base: 512` |
| --- | ---: | ---: |
| width 512 (101M) | 4.801 | (= default) |
| width 1024 (304M) | 4.680 | 4.661 |
| **width 2048 (1.0B)** | **4.702** | **4.564** |

| Paired comparison (document-bootstrap 95% CI) | ΔCE |
| --- | ---: |
| 1B scaled vs 1B default | −0.138 [−0.147, −0.130] |
| 1B default vs 300M default (bigger model **worse**) | +0.022 [+0.015, +0.029] |
| 1B scaled vs 300M scaled | −0.097 [−0.104, −0.092] |
| 1B scaled vs 100M baseline | −0.237 [−0.244, −0.230] |

~~Under the default parameterization, scaling width 300M → 1B *hurts* at the transferred LR.~~ **Correction (seed 202, 07:27):**
the default 1B at LR 4.5e-4 scored 4.652 with seed 202 (vs 4.702 with seed 101), *better* than the 300M default (4.671). The
"1B default worse than 300M at the base LR" result is within seed noise and is withdrawn. The default 1B shows the largest
same-config seed spread so far (0.050).

**Two-seed base-LR result (seed 202 scaled, 09:00): width scaling helps at 1B.** Scaled 1B: 4.564 / 4.565 (seeds 101/202, spread 0.001);
default 1B: 4.702 / 4.652. Paired scaled − default: **−0.138 / −0.087** (mean −0.11). Both seeds agree in sign, and the scaled arm is far more
seed-stable than the default (0.001 vs 0.050 spread). The size of the gain is uncertain (0.05 between seeds); its sign is not. The seed-101 default run also showed a mid-training
loss bump (5.70 → 6.19 between steps 500 and 1500) and MLP-product kurtosis growing to ~10k. With width scaling, the same model improves
monotonically with size (4.801 → 4.661 → 4.564), and its live MLP tails stay ~20× lighter (kurtosis ~100 vs ~2,500 at step 3500)
with lower training loss at every logged step. This is the first evidence that the rule holds over a 4× width ratio *at the base LR*.

**Seed 202, default at LR 1.35e-3 (10:45):** 5.006 (seed 101: 4.996), so the default's high-LR result replicates closely. Its 3×-LR penalty
is +0.354 on seed 202 (+0.294 on seed 101). The scaled seed-202 high-LR run is in progress.

**Seed 202, width-scaled at LR 1.35e-3 (12:18): diverged, CE 5.657** (seed 101: 4.874; default seed 202: 5.006). Max |S1·S2| went
4.9k (step 500) → 887k (step 1000) → 10M (step 2000). The loss spiked 6.33 → 8.2 with grad norm 138 at step ~1500, and the run never
recovered (parent final 5.61). Seed 101 at identical settings stayed at 2k–26k with no spike. **At 1B, LR 1.35e-3 sits at the edge of
stability even with width scaling, and whether a run diverges depends on the seed.** The two-seed high-LR penalty for the scaled arm is
+0.310 / +1.092, so width scaling alone does not make 3× the base LR safe at 4× width. Both default runs at this LR also ran at
1e6–5e7 score maxima without a loss spike but finished worse (5.00/5.01). Consequence: QK-clip (which bounds exactly this quantity) moved to
the front of the queue, with the diverging seed 202 as its first candidate (`runs/queue2.sh`).

**Fixed-WD arm (preliminary, seed 101, LR 4.5e-4; 13:52):** width-scaled LR with WD *not* rescaled (`wsfwd`) scored 4.577, vs 4.564 for
the default rule (WD ∝ width^+½). That's +0.013 in favor of rescaling WD. It's small, but the scaled arm's seed spread so far is 0.001, so it may
be real; it needs the seed-202 fixed-WD run before it's reported as a finding.

**High-LR update (seed 101, LR 1.35e-3):** 1B default 4.996, 1B scaled **4.874**. Scaling still wins (−0.122), but it **no longer
reduces the 3×-LR penalty at 4× width**: +0.294 default vs **+0.310 scaled**. At 2× width the penalty was cut from +0.20 to +0.08.
The default run's attention products ran away (2.7e5 → 1.6e7). The scaled run's stayed ~1000× smaller (2.2e4 at step 2500) but were
still rising, with heavy MLP tails (kurtosis ~1e4). Reading: the ½ exponent flattens the coordinate check only approximately. Its
residual +0.07–0.12 slopes compound over a 4× ratio, so the *safe LR range narrows with width even under the rule*. Before ≥4×
extrapolation, either fit the exponent at three widths (512/1024/2048, √2-spaced LRs) or combine the rule with a stability guard
(QK-clip). Pending: seed 202 and the fixed-WD arm, which will show whether this is seed noise or a WD effect.

## Interpretation diagnostics on a trained model (brief §10)
`tnlm analyze` on the best width-512 pilot checkpoint (12×512 pre-RMS, signed-product attention, 4.8M tokens), CPU,
2 validation probe batches: `runs/analysis/w512_trained_cpu_r2/`. Took 667 s on 24 CPU cores, $0.08.

- **Exact algebra holds on trained weights:** factored vs explicit local tensors max relative error 9e-7, gain folding on real
  hidden states 4e-7, analytic vs autograd normalizer JVPs 6e-8 at every site.
- **Normalization is not a small correction to component importance:** removing a branch with live norms costs much
  less than removing it with denominators clamped. For example, L10.attn costs ΔCE +0.090 live vs +0.374 clamped (normalization-mediated
  −0.284 ± 0.031 doc-SE), and L10.mlp costs +0.032 vs +0.276. The live RMS denominators largely *compensate* for removed
  components, so a content-only weight graph would misjudge which components matter. This is why the brief keeps
  normalizer nodes explicit.
- **The trained pre-RMS model can't run on constant denominators:** replacing all 25 RMS denominators with their
  calibrated means gives non-finite logits (top-1 agreement 0, CE/KL non-finite, flagged as `nonfinite_metrics`). The
  degree-5 attention amplifies scale errors. A polynomial (P-class) model therefore has to be trained as one (lagged
  package), not obtained by freezing a trained N-class model.
- **Low-rank structure grows with depth:** MLP quadratic forms have stable rank ~60–85 in layer 0 and ~18–33 in layer 11.
- **Replicated on the best confirmation model** (12×1024 + width scaling, 50M tokens cooled, seed 202; GPU;
  `runs/analysis/w1024ws_s202_gpu_rw*_r3/`): local tensors exact to 7e-7. Normalization-mediated ablation effects are larger:
  L10.attn +0.21 live vs +0.79 clamped, and **L0.mlp +0.15 live vs +5.93 clamped**. Fixed denominators again give non-finite
  logits (top-1 agreement 0.0004). Stable ranks contrast more sharply: layer 0 ≈ 118, layer 11 ≈ 9.
- **GPU cost:** 50 s for the full diagnostics at 300M (2 probe batches). `--rw-layer-window 4` reduces read/write pairs from 972 to 580 with the same
  maximum overlap (0.702). Wall time is dominated by the ablation passes, not the graph, on GPU.
- **CPU cost (100M):** 667 s, dominated by the read/write graph (972 matrix pairs) and the 48 ablation passes.
  Both grow roughly as L²·d·m and L respectively, so beyond ~300M analysis must run on GPU and the read/write graph needs
  sampling or blocking. That's a requirement for the funded-run analysis budget.

## Open scaling items
**Done on this box:** width transfer (pilot + 50M cooled, 2 seeds), depth and joint width×depth transfer (pilot),
kernel/CE/checkpointing/Flex fast paths, FSDP2 (gloo), presets + memory planner, trained-model analysis (CPU and GPU),
the 320M confirmation data, CUDA determinism.

**GPU queue (this box; ordered; from measured gaps + `docs/literature/scaling_review.md` recommendations):**
1. *Done:* joint 24×1024 at 50M cooled tokens, 2 seeds (above).
2. *Running (since 23:04):* width 2048 (~1B) at 50M: default / width-scaled / width-scaled with **fixed WD**,
   2 LRs × 2 seeds (`configs/width2048/`). This tests the rule over a 4× ratio and separates the LR and WD effects (review rec. 1).
3. *Queued after width 2048:* optimal-LR shift with training length (rec. 4): 12×512 at 50M vs 300M tokens, LR 2.25e-4…9e-4 √2-spaced (`configs/lr_horizon/`).
4. *Queued after lr_horizon:* critical batch size (rec. 5, `runs/critical_batch/run.sh`, CPU-tested): branch a 100M pilot checkpoint at batch 8K/32K/128K tokens with LR × √(batch ratio);
   β2 tuned in {0.95, 0.99} rather than extrapolated.
5. Depth 12→48 at width 512 with full CompleteP (α=1) at 50M, 2 seeds (rec. 3), before any 64-layer run.
6. Late-training coordinate check, at ~1k and ~5k steps, plus an embedding-LR sweep ×{1, 3, 10} (rec. 2).
7. *Queued after critical batch:* QK-clip at 1B, tau=100 every 50 steps (`configs/qkclip1b/`, CPU-tested; `tnlm/qk_clip.py`).
8. *Queued after QK-clip 1B:* width-exponent fit: width-scaled 512/1024/2048 at LR 3.2e-4/6.4e-4/9e-4 (`configs/exponent_fit/`).
9. Stability instrumentation (rec. 7, done: `stability_stats`, live logging, `scripts/stability_trace.py`): per-layer max/RMS of S1, S2, S1·S2 and of the MLP product, on trained checkpoints.
   QK-clip first if scores grow. Z-loss 1e-4 as an ablation.

**Remaining, needs a decision or other hardware:**
1. **Beyond-2× width at 50M tokens:** width 2048 (12 layers, ~1B params), 2 LRs × 2 seeds × {default, scaled}. On this box that's
   ~10 GPU-hours (~$4). Confirms the rule over a 4× ratio before any ≥1B pilot.
2. **Depth and joint rules at 50M cooled tokens** (pilot-only so far): 24×1024 default vs scaled, 2 LRs × 2 seeds, ~6 GPU-hours (~$2.5).
3. **Multi-GPU:** NCCL run of `tests/dist_workers/fsdp_equivalence.py` and the distributed trainer on a ≥2-GPU box (only gloo/CPU verified here).
4. **H100/A100:** re-tune kernel tiles with `python -m tnlm.kernels.tune` (larger shared memory), re-measure memory constants
   for `shapes.py`, and check the 0.74× poly/softmax throughput ratio there.
5. **Batch-size / β2 transfer** for the ≥1B configs (`global_prediction_tokens_per_update`, β2 are not validated; see config headers).

_Retention note (15:25):_ the uncooled parent finals of the 8 default-parameterization confirmation runs were deleted to make
room for the ~1B width-2048 sweep (each 1B candidate stages ~80 GB). All cooled finals, evaluations and comparisons are kept,
as are the 4 width-scaled parents (for continuation). The deleted controls can be regenerated from `configs/confirmation/`.
_Retention note (00:45 Oct 17):_ to keep the 1B sweep from hitting disk admission overnight, the remaining confirmation cooled finals were
deleted except those cited in analyses (w1024ws_lr0.00045_s202, and w1024 / w1024ws at lr1.35e-3 s101) and the 4 width-scaled
parents. All evaluations, comparisons and stability numbers are retained.
