# Readiness for a multi-H100 run (written 2026-10-18 02:10; rental planned in ~2 days)

Goal: when the H100 node arrives, every GPU-hour goes into the main run, not into discovering problems. This lists what is verified on the
single RTX 5000 Ada box, what can only be verified on the rental, and what is still open, with an owner for each.

## 1. Recipe (what the big run should use)

| Decision | Status | Evidence |
| --- | --- | --- |
| Width scaling: LR ∝ width^-½, WD ∝ width^+½ (`width_scaling_base: 512`) | **Verified to 1B** (4× width, 2 seeds) | −0.138 / −0.087 vs default at 1B (docs/scaling.md) |
| Base LR 4.5e-4 at width 512, transferred | **Verified to 1B** | 3× this LR is unsafe at 1B without QK-clip |
| QK-clip τ=100, every 50 steps | **Adopted** in `configs/{confirm_500m,pilot_1b,pilot_3b,funded_7b,funded_32b}.yaml` | Rescues divergence (5.657 → 4.966), free when idle (−0.002 / −0.004) |
| Depth scaling (CompleteP, `depth_residual_base: 12`) | Verified at 24 layers only | Joint 24×1024 sweep; ≥48 layers untested |
| LR vs token horizon | **Done (09:39 Oct 18):** LR_opt ∝ D^−0.10; use ~3.5e-4 base LR (at width 512) for a 28B-token run | docs/scaling.md "LR vs token horizon" |
| Global batch size (critical batch) | **Queued** (`runs/critical_batch/run.sh`) | Matters most on 8+ GPUs (large global batch) |
| **Normalization package** (6 options; brief §1, never run) | **Queued next** (`configs/norm_screen`, 18 runs, ~4.5 h): runs before the batch/exponent experiments, which assume `pre_rms` | If `lagged` (fully polynomial) is close, that's the headline interpretability result |
| Width exponent (is ½ right?) | **Queued** (`configs/exponent_fit`, with QK-clip) | Refines the LR rule at 1B |
| Behaviour at a compute-optimal horizon | **Not yet tested**: all runs ≤ 300M tokens | Plan: 100M model × ~2B tokens here (~8 h) before the rental |

## 2. Software for multi-GPU

| Item | Status | Owner |
| --- | --- | --- |
| FSDP2 sharding, distributed trainer, sharded checkpoint/resume, distributed Muon, calibration | Built, tested with multi-process **gloo on CPU** (in the passing suite) | Claude / Codex |
| Same on **NCCL / real GPUs** (`tests/dist_workers/fsdp_equivalence.py`, distributed trainer) | **Only verifiable on the rental.** Make it the first 10 minutes there | Day-1 runbook (§4) |
| Triton poly-attention tile tuning for H100 (`python -m tnlm.kernels.tune`) | Tuned for sm_89 only; H100 needs a re-tune (larger shared memory) | Day-1 runbook |
| Memory planner (`tnlm.shapes`) on 80 GB GPUs | pilot_1b (1.4B) and pilot_3b (2.8B) fit on 1 GPU; funded_7b needs ≥2, funded_32b ≥16 (8k tokens/GPU) | Re-measure constants on H100 |
| Checkpoint retention for a run whose **weights we keep** | Screening retention (prune everything) is the current default; a long run needs periodic full checkpoints and a keep-best/final policy | Claude: config + doc |

## 3. Data

| Item | Status | Owner |
| --- | --- | --- |
| Tokenized FineWeb | 320M tokens on disk; Chinchilla for the 1.4B pilot needs ~28B (~56 GB at 2 B/token) | |
| `prepare_data` at scale | **Gap.** Single-threaded tiktoken plus in-memory ranking of all candidate documents. Fine at 320M; unproven at 28B (est. ~5 h single-threaded, memory unknown) | **Codex** (requested 02:10) |
| Plan | Make it sharded/parallel, verify here on ~3B tokens (also feeds the 100M validation run), run the full 28B+ prep on the rental's CPUs on day 1 while kernels tune | |

## 4. Day-1 runbook on the rental (draft)
1. `pytest -n auto -m "not slow"`, then the NCCL equivalence worker and a 2-GPU trainer smoke.
2. `python -m tnlm.kernels.tune` on H100; record tiles.
3. Start the full data prep on CPU (in parallel with 2).
4. Canary: `pilot_1b` for ~200 steps on all GPUs → measured tok/s, memory per GPU, loss sane; replace the estimates in §5.
5. Launch the main run with full checkpoints every ~1 h and the live stability watcher.

## 5. Rough size and cost (estimates; assume 35% MFU on H100 bf16, to be replaced by the day-1 canary)

| Config | Params | Chinchilla tokens | 8×H100 est. | Note |
| --- | ---: | ---: | ---: | --- |
| pilot_1b | 1.4B | ~28B | **~1 day** | The natural first big run |
| pilot_3b | 2.8B | ~56B | ~4 days | |
| funded_7b | 6.9B | ~140B | ~3 weeks on 8; ~3 days on 64 | Needs more GPUs |
| funded_32b | 34B | ~690B | needs ≥16 GPUs just to fit | Out of scope for one node |

### Cost estimates (02:20; 6·N·D FLOPs at 35% MFU ≈ 350 TFLOP/s per H100; price per GPU-hour is an assumption)
| Run | Tokens | GPU-h | at $1.5 / $2 / $3 per GPU-h |
| --- | ---: | ---: | --- |
| pilot_1b (1.4B), first-look | 10B | ~70 | $100 / $140 / $210 |
| pilot_1b (1.4B), Chinchilla | 28B | ~190 | $280 / $380 / $570 |
| funded_7b (6.9B), Chinchilla | 140B | ~4,700 (+~33% if activation checkpointing is needed) | $7.0k / $9.3k / $14k |

MFU is the largest uncertainty (25–45% gives ×1.4–×0.8). The day-1 canary measures it. Recommendation: 1.4B first; decide 7B after its curve.

### Updated costs at October-2026 rates (02:35; same FLOP model; H100 index ~$2.82–2.92/GPU-h, list median ~$4.20, Vast.ai $1.47–3.00)
| Run | Params | Tokens | H100 GPU-h | ~$2.8 | $2–4.2 | Wall-clock |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1.4B first look | 1.4B | 10B | ~70 | ~$200 | $140–300 | ~9 h on 8 |
| 1.4B Chinchilla | 1.4B | 28B | ~190 | ~$530 | $380–800 | ~1 day on 8 |
| 7B Chinchilla | 6.9B | 140B | 4.7k–6.3k | ~$13–18k | $9k–26k | ~3 days on 64 |
| 32B Chinchilla | 34B | 690B | 114k–152k | ~$320–425k | $230k–640k | ~19 days on 256; needs multi-node (≥16 GPUs to fit) |

B200 (~2.2× H100 FLOPs) rents at ~$5.9–8.0, so roughly break-even per FLOP; H200 (141 GB) at ~$3.4–5.3 helps 7B memory.

### Compute sources (verify terms directly)
NAIRR (US institution needed; 3-page proposal; start-up allocations reviewed in ~2 weeks), Google Cloud AI Safety Research Credits (≤$50K),
NVIDIA Academic Hardware Grant (AI-safety track, interpretability prioritized), Lambda Research Grant (≤$5K). Not verified in searches:
Open Philanthropy / Coefficient Giving, LTFF, UK AISI grants, MATS, university-cluster collaborations. The project's pitch is
interpretability: an exactly polynomial LM, with the 1B results and a ~$500 1.4B run as proof that it works.

### Checkpoint retention gap (found 02:30; **fixed 02:45**)
With default `milestone_tokens`, a 28B-token run retains **286** full checkpoints (~17 GB each, ~4.7 TB). The scale-up configs need an explicit
sparse milestone list plus rolling recovery checkpoints.
**Fix:** `configs/{pilot_1b,pilot_3b,funded_7b,funded_32b}.yaml` now set log-spaced `milestone_tokens` (1B, 2B, 4B, … 512B) and
`keep_recovery_checkpoints: 2`. Permanent states: 5 / 6 / 8 / 10 milestones (+ initial, branch, final) for the four Chinchilla runs, i.e.
~8 states × 17 GB ≈ 140 GB for the 1.4B run instead of ~4.7 TB.

### Data loader (resolved 05:40 Oct 18, Codex `0b65c6a`)
Shared read-only mmap index (40 bytes/document, prebuilt at prep time) replaces per-rank metadata materialization. At 3B tokens: peak RSS
**0.55 GB (was 11.6 GB)**, cached startup 9.8 s (was 65 s), identical batches/lineage/state, near-end seek+restore 0.06 s. Report:
`docs/datasets/compact_sampler.md`. 30B / 8×H100 RSS is projected (~linear in documents: ~0.4 GB of index per 10B tokens, shared), measure on day 1.
Scalable prep verified at 3B: 2.7M tokens/s, 5.9 GB peak RSS with 8 workers (`docs/datasets/parallel_preparation.md`), so 28B ≈ 3 h on 8 workers.

### Decision (user, 11:00 Oct 18)
The H100 run uses **`pre_rms`** (the only normalization proven stable to 1B with this attention) plus width scaling, QK-clip, the LR-horizon rule
and sparse checkpoints. The fully polynomial normalization is a **separate research track**, not a blocker: every polynomial option diverges in
training at 12 layers (docs/scaling.md "Polynomial normalization canaries"). The norm screen's analysis-cost table will quantify what `pre_rms` costs.

### Recipe-change checklist (adopted 14:50 Oct 18)
Before a component replaces part of the H100 recipe, test in one queue rebuild: (1) loss at the 100M screen with ≥2 seeds; (2) LR optimum
(not at the edge of the grid); (3) compatibility with the other recipe parts (QK-clip, width scaling, compile/kernels); (4) 1B stability on the
seed and LR that diverged before (w2048, seed 202, LR 1.35e-3); (5) analysis cost (`scripts/analyze_kept_finals.sh`). Current candidate:
`spectral_rms`, with (1)–(2) in `configs/norm_followup` and (3)–(4) in `configs/spectral1b`.
