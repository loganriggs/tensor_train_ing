# Tensor network language model research and implementation plan

Prepared for Logan Riggs Smith and the coding agent implementing the experiments. Research cutoff: 15 October 2026. Personal compute and agent API budget: **USD 1,000 total**. Target of the later funded program: dense models up to 32B parameters. This brief is self-contained; no earlier conversation is required.

## 1 Objective and immediate decisions

Build a small, reproducible language-model experiment platform that finds better bilinear architectures and normalization schemes while preserving useful weight-based analysis. Deliver efficient training, exact data lineage, resumable checkpoints, and evidence that selected improvements survive longer training, longer context, and greater width. Do not spend this budget trying to prove a 32B scaling law.

The model family replaces SwiGLU with `D((Lx) * (Rx))` and softmax attention with products of two independently learned QK score matrices. RMSNorm is allowed in the practical family. A separate polynomial family tests whether input-independent normalization can work without unacceptable instability or loss. Keep the distinction explicit in exports and results.

Recommended first implementation:

- A roughly 100M model, context 512, BF16 computation with FP32 norm reductions and optimizer state; a roughly 500M confirmation model.
- Pre-RMSNorm and signed bilinear attention as the reference tensor architecture; conventional SwiGLU plus softmax as the control.
- Six normalization packages, tested cheaply with a small learning-rate sweep. Then a limited architecture search around the best normalization choices.
- AdamW as the initial verified optimizer; Muon plus AdamW as a separately tuned finalist comparison.
- Short extendable trajectories, separate cooldown branches, moderate token batches, and explicit tests of ranking transfer.
- Compiled PyTorch first. A custom attention forward **and backward** kernel only when profiling shows sufficient benefit.
- Weight-defined content connections and explicit normalization dependencies, evaluated separately. No claim that scalar normalization can simply be ignored.

Reuse Logan's existing implementation if available and correct. Otherwise use OLMo-core modules and training infrastructure where practical [R20]. Avoid a major framework migration during the MVP. Pin the code revision, dependencies, tokenizer, data manifest, and kernels. The requested deliverable here is a plan; provisioning paid resources requires the user's chosen account, endpoint, and launch scope. The implementation must enforce the budget once runs are authorized.

## 2 What preserving a tensor network means

Use these categories throughout the code and results:

| Class | Forward computation to logits at a fixed checkpoint | Weight analysis available |
| --- | --- | --- |
| P | Finite polynomial computation over continuous embedding inputs: linear maps, products, sums, fixed masks, position-dependent constants, biases | Exact local coefficient tensors; exact finite polynomial representation in principle |
| N | Class P plus explicit input-dependent normalization nodes | Exact local polynomial factors and exact normalizer functions; conditional tensor analysis with normalizers retained or fixed |
| G | Additional input-dependent nonpolynomial gates, routing, softmax attention, tanh, erf, clipping | Does not meet the chosen polynomial-plus-normalization target |

Class P refers to a fixed inference checkpoint. Training-time optimizers, weight projections, and frozen running statistics need not be polynomial in the weights. Softmax cross-entropy is allowed as the training objective; the representation boundary is the **logits**, not the probability normalization.

A finite vocabulary and bounded sequence length make a gigantic lookup tensor possible for almost any model. That is not the useful structural claim. The goal is a compact contraction structure over embedding coordinates and reusable components.

Neither P nor N automatically guarantees efficient exact Gaussian expectations, globally sparse circuits, or cheap global contraction. Repeated inputs, symmetrization, polynomial degree, and contraction structure matter. A compact forward program can represent an enormous coefficient tensor. The tensor-similarity literature establishes results for specific multilinear structures and metrics; do not apply its theorem to arbitrary normalization-augmented models without checking assumptions [R01, R02].

### Exact weight structure of a bilinear block

With column-vector convention,

\[
B(x)=D((Lx)\odot(Rx))=\sum_h d_h(l_h^\top x)(r_h^\top x),
\qquad T_{oab}=\sum_h D_{oh}L_{ha}R_{hb}.
\]

Because both inputs equal x, only `sym(T)` on the two input axes affects the function. Export low-rank factors and implement tensor contractions without materializing `d_out × d_in × d_in` at model scale. Biases can be represented by an appended constant coordinate. This is the basis of the bilinear interpretability approach [R03].

For an upstream write direction `d_i` and a downstream read vector `l_j`, `l_j^T d_i` measures a direct linear read/write overlap. The downstream bilinear channel also depends on its other input. Report both encoder legs, their interaction, and downstream output vectors. Nonzero overlap is a possible computational connection, not evidence that the component is used, semantically meaningful, or causally important on a dataset.

Factor magnitudes are not invariant to `l_h -> c l_h, r_h -> r_h/c`, permutation, or more general allowable basis changes. Canonicalize factor scales/signs for presentation, compare symmetrized functions where feasible, and accompany raw edge magnitude with normalized overlaps and actual intervention results. Do not force orthogonality merely to make a graph look sparse.

## 3 Normalization and what weights still tell us

### RMSNorm separates a fixed map and a scalar function

For a d-dimensional vector, let

\[
r(x)=\sqrt{x^\top x/d+\epsilon},\qquad N(x)=Gx/r(x),\quad G=\operatorname{diag}(\gamma).
\]

At a fixed checkpoint, G folds into adjacent linear maps. The denominator does not. For a bias-free homogeneous quadratic block,

\[
B(N(x))=\frac{D((LGx)\odot(RGx))}{r(x)^2}.
\]

The quadratic feature directions and local coefficients are weight-defined; a common scalar modulates the branch. This is already useful. It does not turn the whole residual network into the same polynomial with a constant rescaling [R04].

The exact Jacobian is

\[
J_N(x)=\frac{G}{r(x)}-\frac{(Gx)x^\top}{d\,r(x)^3}.
\]

The second term is a rank-one, input-dependent coupling. Even if `l^T G delta = 0`, changing x by delta can affect `l^T N(x)` through the denominator. Therefore absence of a direct content edge does not establish absence of a causal influence.

Implement two edge views:

1. **Content view:** folded read/write overlaps and low-rank polynomial factors.
2. **Normalization view:** explicit normalizer nodes, their scopes, and the scalar sensitivity terms above.

Keep normalization nodes as shared nodes rather than replacing each with an unreadable dense graph. For local sensitivity, compare the analytical full Jacobian with autograd JVPs. For finite interventions, recompute the normalizers.

### Why token mixing makes normalization harder

Suppose the same pre-RMSNorm supplies both QK pairs and V at every position. Fold gains into projections and ignore biases. Then the contribution from token j to token i has the scalar factor

\[
\frac{1}{r(x_i)^2\,r(x_j)^3}.
\]

The query factor is common across source tokens for that head; the key/value factor varies by source token and changes their relative importance. A subsequent output normalization cannot generally eliminate those source-dependent factors. QK normalization adds further per-token, per-head quadratic norm functions. Keep those explicit.

This is a reason to test **attention output normalization as an alternative to additional QK normalizers**. It is an experiment, not an established performance improvement for this architecture. Log attention-score moments by position and head, and test length transfer.

### Homogeneous cancellation has a narrow valid domain

For a nonzero homogeneous map F of degree p, and gain-free zero-epsilon RMS normalization `R0`,

\[
R_0(F(R_0(x)))=R_0(F(x)).
\]

This follows because the inner normalization introduces one positive scalar raised to p. It applies to a single-vector homogeneous branch. Biases, finite epsilon, mixed polynomial degrees, residual addition, and independently normalized source tokens prevent the blanket extension of this identity.

Do not count mathematically equivalent norm placements as distinct representational advances. They can still have different parameterization, numerical, and optimization behavior. Include a tiny test of the identity and counterexamples with a residual and with token mixing.

Homogeneous/projective coordinates can encode some rational maps or scale equivalences. Simply adjoining a coordinate equal to `sqrt(x^T x/d + eps)` does not make that coordinate an independently available polynomial input. Any proposed lifting must specify how the coordinate is computed and which dependency is retained. A valid scale-equivalence result should state whether it preserves directions, logits, probabilities, or full intervention responses; these are different guarantees.

### A focused extension using explicit quadratic norm metrics

An interpretable experimental generalization is `N_M(x) = Gx / sqrt(x^T M x + eps)` with a fixed-at-checkpoint positive-semidefinite metric M. RMSNorm uses `M=I/d`; a head norm after a linear projection depends on a metric proportional to `Q^T Q`; group norms use a separate restricted metric per group. Export these metrics as low-rank factors or diagonal masks alongside the content factors.

Its Jacobian is

\[
J_{N_M}(x)=G/r-(Gx)(Mx)^\top/r^3.
\]

If `M delta = 0`, the denominator is exactly unchanged by `x -> x+t*delta` for every t. This gives a stronger weight-defined statement than merely observing a small sensitivity on sampled inputs. It says nothing about the numerator or other downstream paths.

Two cheap follow-ups are a **fixed subset RMS** and a **low-rank learned norm metric** `M=P^T P/rank(P)`. Partial RMS estimation has precedent [R04]; the learned metric and its proposed use for interpretable dependency control are hypotheses in this plan. A low-rank norm does not control growth in its nullspace. Adding `lambda*I/d` repairs full-space control but restores norm dependence on every direction. Record that tradeoff; do not claim a small projection faithfully approximates RMS without measuring it. Give this experiment a small Stage C slot only if the first norm screen leaves room.

### Candidate normalization methods

| Method | Class | Interpretation benefit and main issue | Priority |
| --- | --- | --- | --- |
| Pre-RMSNorm | N | Stable reference; one scalar dependency per token and branch input | Required |
| Branch-output RMSNorm | N | Polynomial branch before one output scalar; need to control large intermediate products | Required |
| Sandwich pre/output RMSNorm | N | Tests stability; partly redundant for homogeneous single-vector MLPs, but not generally attention | Required |
| Group RMSNorm over four fixed residual groups | N | Denominator-mediated dependencies confined within each group; more gates and a privileged grouping | Required exploratory |
| RMSNorm plus weight spectral reparameterization | N | Controls linear-map scale without introducing extra input gates | Required exploratory |
| Lagged running-RMS scaling, frozen for evaluation | P at a fixed checkpoint if all norms are frozen | Entire denominator becomes a constant; train/eval drift and statistical-state management are the risk | Required exploratory |
| Scalar-gain RMSNorm / ScaleNorm | N | Simpler gain parameterization; does not eliminate radial coupling | Cheap follow-up |
| Fixed subset RMS or low-rank quadratic norm metric | N | Explicit, potentially small norm-dependency subspace; uncontrolled nullspace is the risk | Cheap follow-up |
| ReZero or small fixed residual scales with no activation norms | P | Cleanest polynomial computation; long-run and depth stability unestablished here | Follow-up if lagged scaling is promising |
| LayerNorm | N | Mean subtraction is a fixed projection that can be folded; variance denominator remains dynamic | Optional control |
| SpanNorm / DeepNorm adapted to RMS | N | Tests residual topology and depth stability; does not remove normalization dependencies | Later, after basic placements |
| nGPT | N/G in the complete published architecture | Weight constraints are compatible; activation normalization, softmax, and other components require separation | Borrow individual ideas only |
| DyT, Derf, sigmoid attention gates, dynamic mHC | G | Remove or change one operation by adding other nonpolynomial gates | Exclude from the target family |

RMSNorm and ScaleNorm motivate the first family [R04, R05]. ReZero demonstrates learnable input-independent residual scales [R06]. sigmaReparam normalizes weights spectrally and learns a scalar; its published softmax-entropy rationale does not establish bilinear stability, but its reparameterization is structurally compatible [R07]. χ-net experiments used running RMS batch normalization that was folded at inference on image tasks [R02]. This supports a small frozen-statistics experiment, not a claim of successful LLM scaling.

SpanNorm changes the second residual skip to use the original block input, rather than merely moving a norm; copy its equations accurately if testing it. DeepNorm combines residual scaling with initialization. Neither has a directly transferable stability proof for degree-five polynomial attention [R08, R09]. DyT and Derf use tanh and erf, respectively, so “normalization-free” does not imply polynomial [R10, R11]. nGPT's step-efficiency claims should not be treated as GPU-hour savings for this model [R12].

### Exact implementation of the first six packages

Let F be either the attention branch or the bilinear MLP. Apply packages consistently to both unless a config explicitly names one branch.

```python
# Each occurrence of norm denotes its own configured module.
pre:       x + alpha * F(norm_in(x))
branch:    x + alpha * norm_out(F(x))
sandwich:  x + alpha * norm_out(F(norm_in(x)))
group:     x + alpha * F(group_rms(x, groups=4))
spectral:  x + alpha * F_spectral(norm_in(x))
lagged:    x + alpha * F(fixed_scale_from_previous_steps(x))
```

Use a final RMSNorm in the five N packages. Use frozen running scaling at the final output in the P package; otherwise it must be labeled N. Initial gain vectors are ones, with no norm bias; epsilon starts at `1e-6`. Residual multiplier alpha starts at 1 when the output projection is already depth-scaled. Do not accidentally apply two independent depth-scaling conventions. In the norm-free follow-up, compare a small residual multiplier or ReZero, without also zero-initializing the branch output matrix.

For lagged scaling, estimate statistics from training data only. The current forward pass uses the previous optimizer step's scalar statistic, detached from the computation graph. Accumulate new sufficient statistics outside the compiled/recomputed branch and update exactly once per optimizer step, not once per microbatch or activation recomputation. Use a token-based EMA timescale and log its value. Calibrate initial constants on a separate training-only prefix, record it as calibration exposure, and evaluate with frozen statistics. No current-batch or future-position statistics may influence an earlier causal output. A frozen, permanently calibrated version and a lagged-updating version are separate experiments.

The initial lagged implementation stores one scalar mean-square statistic per norm site, averaging over valid tokens and features. Start with 100,000 calibration tokens and an EMA half-life of 1,000,000 tokens; these are proposed tunable settings. Normalize by `sqrt(previous_mean_square + eps)`, update with count-weighted sufficient statistics, and exclude padding. Stop a run on nonfinite values rather than silently clipping the activations into a different model family. Calibration cost and exposure belong in the ledger and data manifest.

For spectral reparameterization, export the effective matrices actually used, not only the underlying unconstrained parameters. Save any power-iteration state. Freezing the effective matrices must reproduce the checkpoint's logits. Spectral control bounds map norms, not all polynomial activations globally: nonconstant polynomials remain unbounded on an unbounded input domain.

For group RMS, groups are fixed contiguous feature partitions, not token or batch partitions. No group shuffling between layers in the first experiment. Recompute the block-diagonal normalizer Jacobian and report the performance cost of this imposed locality. Grouping alone does not make the dense linear maps sparse.

## 4 Architecture experiments after normalization screening

Use `S1 = Q1 K1^T / sqrt(dk)` and `S2 = Q2 K2^T / sqrt(dk)`, followed by a causal/document mask. The default numerator is `S1 * S2`; negative scores are intentional. The product is degree four in embeddings and degree five after multiplying V, even though each score is bilinear.

| Variant | Exact change | Reason to test | Cost and interpretation |
| --- | --- | --- | --- |
| Signed product | `A = S1 * S2` | Existing reference | Two QK pairs; retain signed interactions |
| Offset square | `A = (S + b)^2`, initial `b=1` | Nonnegative attention plus lower-degree terms | One QK pair; reallocate saved parameters for a matched-budget comparison |
| Lower-degree mixture | `A = c0 + c1*S1 + c2*S2 + c12*S1*S2` | Avoid forcing every attention contribution through a product | Coefficients are learned input-independent head scalars; start `(0,0,0,1)` |
| Affine bilinear MLP | `D((Lx+bL)*(Rx+bR))+bD` | Include linear and constant terms explicitly | Small parameter change; breaks homogeneous cancellation |
| Smaller QK dimension | `dk=32` versus `64`, with value width unchanged | Reduce recurrent state and QK arithmetic | Changes capacity; profile and compare token efficiency |
| Fixed local/global pattern | Fixed windows in some layers, full polynomial attention in others | Reduce attention cost at longer context | Position-defined masks preserve P/N classification; defer long-context claims |
| Static residual mixing or skips | Learned input-independent scalars/matrices across layers | Reusable paths without routing gates | More edges, not automatically simpler interpretation |
| Shared or embedding-only V | Share a value map or read values from initial embeddings | Simplifies some cross-layer analysis | Material architecture restriction; funded follow-up |
| Block-term or sparse Tucker bilinear cores | Structured interactions beyond channelwise CP factors | Explicit groups or reusable subspaces | New kernels and rank selection; funded follow-up |

For the MVP, test the first four rows and one `dk` comparison. Do not cross every variant with every normalizer and optimizer. A learned low-rank attention feature map, higher polynomial powers, and polynomial output gates remain possible, but their degree/state costs must be measured before inclusion. Polynomial gates are not guaranteed substitutes for the benefits of sigmoid gates.

Attention magnitude needs explicit semantics. For raw signed attention, start with a position-only factor `n_visible^-1/2`; for nonnegative squared attention start with `n_visible^-1`. These are initialization heuristics under different mean assumptions, not universal scaling laws. With gain-free output RMS and negligible epsilon, a positive common row factor cancels. Do not let a numerically redundant scale become a purported architecture improvement. Profile score/output moments and sweep a fixed head scale if needed. Never divide signed attention by its signed row sum: cancellation makes that denominator unstable.

Squared scores also lose the sign of an unshifted dot product. The offset-square experiment should log b and compare against b fixed at zero if the result is promising. Positivity and additional low-degree terms are separate hypotheses.

For every changed architecture, record parameters, measured tokens/sec, memory, and forward/backward FLOPs where available. Use both matched-parameter comparisons and a smaller fixed-shape diagnostic to reveal gains caused by reallocating MLP capacity. Equal parameters do not imply equal compute.

## 5 Training literature and decisions it changes

| Literature | What it establishes | Decision in this project |
| --- | --- | --- |
| μP / Tensor Programs V [R13] | Hyperparameter transfer across width under a particular parameterization | Add coordinate checks; hold head dimension fixed when testing width transfer |
| CompleteP [R14] | Depth-aware parameterization and feature-learning benefits in Transformer experiments | Test depth separately; do not copy residual exponents without checking the bilinear model |
| CompleteᵈP [R15] | Transfer rules spanning width, depth, batch, duration and per-module tuning | Start with coarse parameter groups; distinguish width transfer from horizon transfer |
| Optimal LR across token horizons [R16] | Longer training can require lower peak LR | Retune/transfer LR for finalists; do not extrapolate a 100M-token winner unchanged |
| Constant LR plus cooldowns [R17] | Reusable training trajectories can support multiple training horizons | Branch cooldowns and continue the uncooled parent |
| Small-batch LM training [R18] | Small batches can work when optimizer timescales are adjusted | Benchmark moderate batches; avoid accumulation solely to imitate large-cluster recipes |
| Resolving compute-optimal discrepancies [R19] | Output-layer cost, warmup, and scale-dependent tuning affect scaling conclusions | Count unembedding cost; avoid excessively long warmup in tiny runs |
| Fair optimizer comparisons [R21] | Rankings can change with tuning, model scale, and completed schedules | Equal tuning budgets and cooldown comparisons for AdamW and Muon |
| Large-scale Muon/SOAP [R22] | Matrix optimizers can work at large training scales | Muon is a candidate, not an assumed 2× speedup; defer SOAP initially |
| Hyperband [R23] | Adaptive resource allocation improves search efficiency in its tested settings | Successive promotion with a few retained weak candidates to detect ranking failures |
| OlmPool [R24] | Short-context metrics can miss long-context architectural differences | Direct context checks for attention/norm finalists |
| Cut Cross-Entropy [R25] | The unembedding/loss can dominate small-model memory; fused loss avoids full logits | Inspect this bottleneck before spending all effort on attention |

These are methods and evidence from other model families, not proofs of transfer to bilinear attention. The search itself must test that transfer. CompleteᵈP's reported experiments include 50M/1.6B-token tuning; this does not validate architectural selection after only 100M tokens [R15].

### Starter recipe and search variables

Default to AdamW while establishing reference correctness. If the existing implementation has a validated optimizer recipe, retain it as a named control. Otherwise start with LR candidates `1.5e-4, 4.5e-4, 1.35e-3`, beta1 `0.9`, weight decay `0.1` on hidden matrices, and gradient clipping at global norm `1.0`. Exclude norm gains, biases, and residual scalars from decay. Declare the embedding/unembedding decay rule explicitly and keep it fixed within a comparison. These numbers initialize the search; they are not inferred 32B settings.

For a fresh fallback implementation, make initialization explicit: independent Gaussian embedding entries with standard deviation 1; Q/K/V and L/R entries with standard deviation `fan_in^-1/2`; D/O entries with standard deviation `(2*layers*fan_in)^-1/2`; unembedding entries with standard deviation 0.02; zero biases, unit norm gains, alpha=1. Use RoPE base 10,000 on both Q/K pairs. This is a proposed conventional starting parameterization, not μP. Before any sweep, inspect branch/residual RMS and logits on training-only calibration inputs and reduce an excessive initial output scale consistently across the compared family. Retain the validated existing initialization as the preferred reference when available. A qk_norm follow-up uses per-head RMS over dk, then RoPE; specify whether gains are shared across heads.

Use a nominal global batch of 8,192 valid prediction tokens when it fits efficiently. Benchmark 2,048 and 8,192; use the smaller efficient batch if larger batches require expensive accumulation without a clear quality benefit. Fix the selected token batch during the initial architecture comparison. Changing sequence length changes sequences per update, not silently tokens per update.

With variable document/loss masks, normalize accumulated gradients by the total number of valid targets in the update, not by the number of microbatches. The sampler must report the actual count. Keep reference parameters and gradients in FP32 under BF16 autocast, with FP32 Adam moments; treat a lower-memory parameter representation as a separately verified implementation change. Start without a z-loss; if logit growth motivates one, test it as an explicit shared training ablation and report the original CE separately.

If beta2 is 0.95 at 32,768 tokens/update, preserving its token half-life gives

\[
\beta_{2,new}=\beta_{2,ref}^{B_{new}/B_{ref}},
\]

or approximately 0.98726 at 8,192 tokens/update. This is a timescale-preservation identity; it is not a full optimizer-transfer rule. Learning rate, epsilon, momentum, and decay may also need adjustment [R18, R15]. Log cumulative LR and weight-decay exposure so duration changes are not mistaken for equivalent regularization.

Start warmup at 1% of a rung's token budget, with a floor of 100 optimizer steps. Flag runs where warmup consumes more than 10% of the rung; increase the rung or reduce the floor rather than comparing mostly-warmup runs. Use constant LR thereafter. For a rung with total endpoint D, branch a linear cooldown at 0.9D and run it for 0.1D; preserve an uncooled parent at the same token position for continuation. Count the extra branch compute and dataset exposure. Rank cooled endpoints together, never cooled versus uncooled checkpoints. This 10% fraction is a starting heuristic; check the finalists against a full cosine run if rankings are close.

Warmup occurs once at the initial run start. Promotion continues the saved uncooled trajectory without restarting warmup or retroactively changing its history. The cooldown branch does not warm up again. A newly trained longer-horizon run with a transferred LR is a new experiment, not a resumed version of an incompatible schedule.

For Muon, use an established implementation with its documented matrix-size update scaling and weight decay. Apply it to hidden matrices, including D and attention O; use AdamW for embedding, unembedding, gains, biases, and scalar parameters. Give each optimizer its own three-point LR calibration around a sensible update-RMS scale. Weight-only spectral normalization and norm placement can interact with the optimizer, so repeat the best two normalization packages under both optimizers before selecting a winner. Keep optimizer differences out of the initial normalization ranking.

Do not label a parameterization μP merely because one learning rate scales with width. Check activation RMS and changes after one/few optimization steps across widths at fixed depth and head dimension. Confirm LR minima across two widths. For the MVP, an explicit conventional parameterization plus a small retune is acceptable; deriving the full bilinear CompleteᵈP extension is a separate deliverable, not a dependency that blocks all experiments.

## 6 Search schedule and dollar limits

All limits below include rented GPU time, compilation, evaluation, checkpoint writes, and separately metered agent API use. They are spending ceilings, not predicted costs. Stop adding experiments when measured throughput predicts the remaining obligations will exhaust the envelope.

| Stage | Work | Cap |
| --- | --- | ---: |
| A | Reference correctness, causal tests, profiling, one bounded kernel attempt | $150 |
| B | Six normalization packages × three LRs, about 100M tokens each at ~100M parameters | $175 |
| C | Extend four normalization finalists to 300M tokens; focused architecture screen with at most twelve extra short runs | $175 |
| D | Optimizer check and width/context confirmation, including a conventional control and replication where needed | $350 |
| Reserve | Failed runs, storage, agent charges, slow configurations, final reproducibility checks | $150 |
| Total | Hard project ceiling | $1,000 |

Stage B may begin at 256 tokens for a throughput-oriented pilot, but use 512 for the main screening table. Each normalizer receives equal LR-search budget. Keep a strict-polynomial candidate alive if it is stable and close on validation loss, even if it is not the raw loss winner. Search explores a quality–interpretability–cost frontier.

Stage C: retain the two best packages, the best stable polynomial package, and one middling package. If polynomial normalization fails, retain another RMS variant instead. Measure ranking agreement from 100M to 300M tokens; close ties remain ties. For architecture candidates, vary one feature at a time around one or two surviving packages, then test a combined winner. Do not present an additive sum of independently measured gains as a verified combined gain.

Stage D targets roughly 0.5B–1B tokens on small finalists and 1B–3B tokens for the ~500M tensor/control pair **only if the measured budget permits**. A valuable cheaper fallback is two widths at fixed depth plus a 2K context check. Changing both width and depth at once is a scale confirmation, not an isolated μP test. Use a context ladder on the same trained checkpoints and short continued-training branches at 2K, plus a matched early-from-scratch 2K run for the most important architectural comparison. This distinguishes extension difficulty from a true context-dependent training ranking.

Use a fixed validation set of at least one million tokens for frequent measurements and a separate untouched final test set. Match context and packing when comparing CE. For cross-context evaluations, score a common set of target positions with their explicitly specified available prefixes. Do not compare losses on differently weighted positions as though they were identical tests. Bootstrap documents or packed sequences, not individual correlated tokens. Finalists need a second seed if improvements are small. A confidence interval across documents is not a confidence interval across training seeds.

Promotion requires finite training, useful held-out improvement, manageable throughput, and an intact interpretation export. A short-run loss winner that fails the normalization sensitivity or causal checks is not a successful interpretable architecture.

## 7 GPU requirements and realistic accounting

### Model shapes

If Logan's existing tokenizer is available, use it and recompute exact counts. A concrete fallback is the GPT-2 tokenizer, padded vocabulary 50,304, with untied embedding and unembedding. Mask padded vocabulary logits out of CE and sampling. Pin tokenizer bytes and EOS ID. The following counts exclude small norm/bias parameters and assume no GQA, `heads * dv = d_model` and initially `dk=dv`.

| Configuration | Layers | Width | Heads | Head dim | MLP width softmax / polynomial | Approximate parameters softmax / polynomial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Screen | 12 | 512 | 8 | 64 | 2048 / 1664 | 101.84M / 101.06M |
| Confirmation | 24 | 1024 | 16 | 64 | 4096 / 3392 | 505.68M / 504.10M |

The smaller polynomial MLP compensates approximately for two extra QK projections. Squared-attention variants need a fresh parameter-count match. Preserve a fixed-shape comparison so redistribution of parameters is visible. For isolated width transfer, create a 12-layer, width-1024 intermediate model; do not use the 24-layer row as the sole test.

### Hardware plan

| Workload | Practical starting hardware | Initial microbatch and memory strategy |
| --- | --- | --- |
| CPU algebra/unit tests | CPU, 16GB RAM | Tiny dimensions; FP64 reference |
| ~100M screening, T=256–512 | One NVIDIA GPU with 24GB VRAM; 48–80GB is easier | Start at 4 sequences, increase to throughput plateau; BF16, bounded logit memory |
| ~500M confirmation, T=512–2048 | Prefer one 80GB A100/H100; 24GB possible with reduced microbatch and checkpointing | Start at 1–2 sequences on 24GB or 2–8 on 80GB; fused attention or checkpointed reference |
| ~3B funded pilot | One 80GB device for small microbatches, or several GPUs | Roughly 48GB model/Adam state before activations under a 16-byte estimate |
| ~7B funded run | 4–8 × 80GB GPUs is a practical starting node configuration | Sharded optimizer/parameters and activation checkpointing; memory feasibility can require fewer GPUs than throughput goals |
| ~32B funded run | Start capacity planning at 16–32 × 80GB; hundreds for fast large-token runs | FSDP/HSDP or tensor/pipeline parallelism; fast interconnect; benchmark the custom attention |

These are capacity-planning recommendations, not measured fits. H100 SXM has 80GB and RTX 4090 has 24GB; they have different bandwidth and kernel support [R29, R30]. No multi-GPU system is required for the $1k MVP. Do not rent eight H100s merely because the old script used eight GPUs.

Use Linux, an Ampere-or-newer CUDA GPU for BF16, a pinned working PyTorch/CUDA/Triton environment, roughly 32–64GB host RAM, 8+ CPU cores for tokenization/prefetch, and 100–200GB local SSD space for the MVP's subset, caches, and checkpoints. If only 24GB is available, lower microbatch before changing the experimental global token batch. Record accumulation overhead. Framework and custom kernel compatibility must be checked on the actual target GPU.

For conventional Adam-style training, budget roughly 16 bytes/parameter for parameter, gradient, and optimizer state, plus casts/workspaces/activations. This is about 1.6GB at 100M, 8GB at 500M, 112GB at 7B, and 512GB at 32B. Actual optimizer implementations differ. An 8×80GB machine has little headroom at 32B after sharding these states; treat that as a tight feasibility experiment, not the recommended production configuration. FSDP shards states but adds transient all-gather memory and communication [R31].

Two full BF16 score matrices alone use `4 * B * H * T^2` bytes. At B=4, H=16, T=2048 this is about 1.07GB **per layer**, before products and backward state. This is why a naive dense-attention implementation can invalidate a seemingly comfortable parameter-memory estimate.

Full FP32 logits use `4 * microbatch_tokens * vocab_size` bytes: 8,192 tokens × 50,304 entries is about 1.65GB. Fused linear cross-entropy or a verified recomputing/chunked implementation can be more valuable than another attention optimization [R25]. A Python loop that retains every chunk's autograd intermediates does not automatically solve this memory problem. If using CCE, explicitly inspect and record any gradient-filtering approximation; start with its exact/non-filtering path where supported, or use another validated implementation.

### Cost controller

At the previously checked public H100 reference rate of $4.50/GPU-hour [R32], $1,000 purchases at most 222 single-GPU hours before other charges. A $150 stage buys at most 33 hours. Use actual provider billing units and minimum reservations; a per-GPU quote does not establish availability of one GPU or a suitable multi-node interconnect.

For measured aggregate throughput q valid prediction tokens/second:

\[
\text{predicted cost}=\frac{\text{remaining tokens}}{q\,3600}\times\text{instance dollars/hour}.
\]

Add explicit compilation, evaluation, storage, and API allowances. Do not assume 300 effective TFLOP/s on small models or novel kernels. Record both steady-state training throughput and end-to-end billed cost. The old `6ND` estimates are only rough planning arithmetic; embedding/output-layer cost and attention can change them materially [R19].

Maintain a persistent spend ledger across restarts. Before every launch, reserve projected cost plus a margin. Stop scheduling when committed plus spent cost reaches $850 unless explicitly consuming the reserve for recovery or final verification. Save and terminate before exhausting the total $1,000 cap.

## 8 Kernel and coding agent work

The preferred optimization order is profiling → compiled reference → existing kernels → one targeted custom kernel. Keep cuBLAS or equivalent GEMMs where they are fast. Do not ask an agent to replace every operation.

The bilinear MLP needs two input projections and one output projection. Fuse the two input projections into a concatenated GEMM when layout and compiler behavior permit; preserve separate logical L/R parameters and optimizer semantics. In particular, applying Muon to a concatenated matrix is generally not equivalent to applying it separately to L and R. Kernel fusion must not silently change the optimizer. Fuse the elementwise product and suitable surrounding operations only after checking backward memory and speed.

For polynomial attention, implement a tiled causal kernel that evaluates S1 and S2 on tiles, multiplies them, applies zero masking and the configured fixed scale, and accumulates into V. There is no row softmax. Triton's fused-attention tutorial is a useful tiling/backward reference, not an implementation to copy unchanged [R26]. FlexAttention's score modifiers still feed a softmax, so `score_mod(score**2)` does not implement this architecture [R27].

Mask excluded entries to zero **after** forming the polynomial, especially when it contains constant or bias terms. Do not reuse the softmax convention of filling scores with negative infinity. Count only causally visible tokens in the same document for `n_visible`; define empty rows explicitly as zero output.

For an upstream output gradient U, the reference backward is:

\[
P=C\odot S_1\odot S_2,\quad H=C\odot(UV^\top),\quad
\nabla_V=P^\top U,\quad
\nabla_{S_1}=H\odot S_2,\quad
\nabla_{S_2}=H\odot S_1.
\]

Here C is the fixed causal/document mask multiplied by the configured row scale; this places the mask/scale exactly once in each derivative. Differentiate each `QK^T/sqrt(dk)` normally, retaining its scaling factor. Save/recompute tiles without materializing global score matrices. Handle RoPE and normalization in separate verified operations initially. FP32 accumulation, BF16 cast points, and score-product precision must be explicit. Check overflowing products before downcasting, not just final logits.

The exact causal recurrent form is

\[
S_t=S_{t-1}+(k_{1t}\otimes k_{2t})v_t^\top,
\quad y_t=(q_{1t}\otimes q_{2t})^\top S_t.
\]

It costs roughly `O(T*dk^2*dv)` with `dk^2*dv` state entries per head. At dk=dv=64 that is 262,144 entries per head; at 128 it is 2,097,152. At short contexts, a tiled quadratic path is usually the first implementation to benchmark. Chunked recurrent training must avoid storing all T states. Manifest's retention repository contains related symmetric-power kernels but its normalization, optional gates, symmetric features, and backward must be checked for equivalence to the selected operation [R28]. Reset recurrent state at document boundaries. A sliding window recurrence needs removal of expired contributions.

Agent-assisted optimization is a bounded experiment. KernelAgent provides a profile–generate–verify–benchmark loop [R33]. KernelOPT reports smaller gains on model-level workloads than isolated operators [R34]. Cap API spend, attempts, and wall time. Give the agent immutable reference code and held-out tests; do not accept speedups obtained by editing tolerances, changing masks, skipping backward, using CPU copies, or caching example answers.

Acceptance requirements:

- Validate Q1/K1/Q2/K2/V gradients, not only forward outputs.
- Test variable lengths, non-tile multiples, batch sizes, head sizes, document masks, zero and large inputs, and non-contiguous layouts if supported. Reject unsupported layouts explicitly.
- Check FP64 finite differences on a tiny reference; compare FP32 kernels tightly and BF16 relative/absolute error against the distribution of reference magnitudes. For near-zero values, absolute error matters. Fix tolerances before optimization.
- Run a 100-step paired training comparison and inspect loss, gradient norms, and parameter updates. Chaotic long-run bitwise identity is not the criterion.
- Time forward+backward+optimizer after warmup with GPU synchronization/CUDA events. Include compilation in project cost. Use the same device, dtype, shape, and math settings for comparisons.
- Add the kernel only if it materially reduces total training-step time or peak memory. A 2× attention speedup when attention takes 60% of runtime yields only 1.43× total speedup.

Do not make a custom kernel a prerequisite for every small screening run. Compiled PyTorch at T=256–512 may already be cheap enough. Compare kernel development/API cost against projected remaining GPU savings.

## 9 Data and checkpoint provenance

For continuity with the existing experiments, use one fixed FineWeb subset if its raw document mapping can be recovered. Otherwise prepare a versioned subset from the official FineWeb release [R35]. Dolma 3 is a strong later option with published training manifests [R20]. Do not change mixture, tokenizer, or preprocessing during the normalization search. An existing token-only cache with unknown document lineage does not satisfy this project's provenance goal.

Define dataset membership using a deterministic hash of stable document IDs, not a streaming `take(N)` over a mutable source. Freeze source revision, shard list/checksums, text preprocessing, deduplication policy, split membership, tokenizer bytes, EOS insertion, packing, and target-loss masks. Store document ID plus original-text/content hash and processed-text hash; record offsets from each packed token range back to source documents. Keep training, validation, final test, and calibration exposures distinct.

Use document-boundary causal masks and reset RoPE positions within documents in the initial design. Boundary transitions must have an explicit loss policy. This costs implementation work but avoids relying on accidental cross-document context. If existing code uses EOS-separated cross-document attention, retain it only as a separately labeled packing policy, never mix policies across compared models.

Checkpoint state includes model, optimizer, LR scheduler, AMP state if any, RNG states, sampler/global example cursor, normalization EMA and spectral-power-iteration buffers, global tokens/steps, skipped-update records, data-manifest hash, config, code commit, environment, kernel revision, and parent-run/checkpoint identity. Save an atomic manifest only after all referenced files are complete and checksummed. Use a recovery scheme that cannot mistake a partial asynchronous save for a valid checkpoint.

Save initialization; early checkpoints at approximately 1, 2, 4, 8, 16, 32 million tokens; then every 100M tokens for short runs. Retain two rolling full recovery checkpoints at a wall-clock interval chosen from measured write overhead. Preserve full state at branch points and final milestones. Record cooldown branches as separate trajectories with their own data exposure. At 100M parameters, a BF16 weight checkpoint is about 200MB; at 500M it is about 1GB. Checkpoints are cheap enough to keep many, but avoid uploading a full optimizer checkpoint after every tiny rung.

Prove replay by hashing the next several packed batches before and after interruption. Exact batch order is required. Bitwise weights are required only for a deterministic tiny reference on identical hardware/software; normal GPU restarts use stated numerical tolerance. After changing world size, reconstruct the global sample stream and document changed accumulation/reduction order rather than promising bitwise-identical training [R31].

## 10 Interpretation evaluation and required outputs

Build these diagnostics before selecting the final normalization scheme:

1. **Local coefficient reconstruction:** compare factored bilinear computation to its explicit symmetrized tensor at tiny width; test biases through homogeneous coordinates.
2. **Read/write graph:** export top signed overlaps for both encoder legs, associated write directions, and norm scopes. Include normalized and raw values; distinguish direct residual paths.
3. **Normalizer sensitivity:** verify analytical JVPs; measure direct and denominator-mediated effects separately. Log norm quantiles by layer/head/position, not just average norms.
4. **Finite ablation decomposition:** evaluate a component removal with all norms live, and separately with norm values clamped to the original forward pass. The difference measures a selected normalization-mediated intervention effect. It is intervention-specific, not a unique causal decomposition.
5. **Fixed-denominator approximation:** compare live logits with frozen-statistic logits using held-out CE, KL, absolute error, and task performance; test longer contexts and shifted domains. A small average error does not establish OOD equivalence.
6. **Low-rank structure:** inspect symmetrized per-output or selected logit-difference quadratic forms, stable effective ranks, and shared input/output subspaces. Compare across seeds and functionally equivalent rescalings. Avoid materializing the global high-degree tensor.
7. **Simple known-mechanism tasks:** a small Boolean hierarchy/XOR task and associative retrieval task, with exact labels and held-out compositions. Use these to test component removal/reuse, not as substitutes for language-model quality.

Do not rank “interpretability” by the number of small raw weights, by a single visually attractive factorization, or by denominator variance alone. A nearly constant denominator can still correlate strongly with the numerator. Report the actual full-minus-clamped response and uncertainty.

Final result bundle from the coding agent:

- Loss versus tokens, versus steady-state GPU-hours, and versus total billed dollars; all counts include cooldown branches where relevant.
- Ranking transfer across duration, width, and context; paired seeds for close finalists.
- A quality–cost–interpretation table with P/N labels and exact normalizer scopes.
- Normalization sensitivity and low-rank/edge diagnostics on fixed probes.
- Effective parameter counts, model shape, precision, kernel version, memory peaks, and throughput.
- Immutable data manifest, checkpoints, reproducible commands/configs, budget ledger, and short experiment log.

The funding pitch should claim a validated small-scale recipe and a reproducible research asset. A 100M-token proxy is not a trained competitive base model, and a successful 500M model does not prove 32B scaling. A slower model can still be valuable if it provides a demonstrable interpretation capability; report that tradeoff rather than applying one arbitrary speed threshold.

## 11 Coding agent implementation contract

Implement the smallest extension to the existing repository that supports this plan. Read repository instructions before edits. Use configuration-driven switches and one authoritative attention reference. Suggested module responsibilities, not a mandate to reorganize working code:

```text
model              bilinear MLP, attention variants, norm placement, parameter counting
normalization      RMS, group RMS, lagged/frozen scalar, spectral effective-weight export
attention          compiled reference, optional fused backend, exact masks
training           optimizer groups, token scheduler, cooldown forks, spend controller
data               immutable manifests, packing indices, document/token mappings
checkpoint         atomic model/optimizer/data/RNG/buffer state and replay
analysis           local tensors, read/write graphs, normalizer JVPs, interventions
experiments        short search, promotion, width/context confirmation
```

The following is the intended config contract. It is not an assertion that these CLI options already exist:

```yaml
project:
  total_budget_usd: 1000
  reserve_usd: 150
  dry_run: true
  spend_ledger: runs/spend.json
model:
  layers: 12
  d_model: 512
  heads: 8
  d_key: 64
  d_value: 64
  d_mlp: 1664
  padded_vocab: 50304
  tie_embeddings: false
  mlp: bilinear
  attention: signed_product
  normalization: pre_rms
  final_normalization: rms
  rms_eps: 1.0e-6
  qk_norm: false
  dropout: 0.0
  position_encoding: rope
  rope_base: 10000
  initialization: explicit_fan_in_with_depth_scaled_outputs
data:
  manifest: REQUIRED_IMMUTABLE_MANIFEST
  tokenizer: REQUIRED_PINNED_TOKENIZER
  context: 512
  document_boundary_mask: true
  reset_position_at_document: true
train:
  optimizer: adamw
  lr: 4.5e-4
  beta1: 0.9
  beta2: 0.9872585449
  weight_decay_hidden: 0.1
  grad_clip: 1.0
  global_prediction_tokens_per_update: 8192
  rung_total_tokens: 100000000
  schedule: constant_with_separate_cooldown
  cooldown_fraction: 0.1
  bf16_compute: true
  fp32_norm_reductions: true
  fp32_optimizer_state: true
  compile: true
  seed: 101
evaluation:
  validation_manifest: REQUIRED_VALIDATION_MANIFEST
  final_test_manifest: REQUIRED_UNTOUCHED_TEST_MANIFEST
  normalizer_probes: true
  causal_prefix_test: true
  resume_batch_hash_test: true
```

For a lagged/frozen run, override **all** activation norm sites, including the final norm, and enforce the P classification via an operator audit. For qk_norm experiments, specify head scope, epsilon, gains, and RoPE ordering rather than a Boolean alone. Conventional controls must enable their actual SwiGLU/softmax operators and be labeled G/control.

Implement and expose these commands or equivalent functions:

```text
validate_config     print exact parameters, operator classification and estimated memory
prepare_data        freeze a bounded data subset and document mappings
check_math          run CPU tensor/Jacobian/causality tests
profile             benchmark one candidate on the supplied GPU
plan_search         output every proposed run and projected cost without launching
train               enforce run and project caps, checkpoint and resume
cooldown            fork from an explicit full-state checkpoint
analyze             produce the interpretation diagnostics
summarize           generate the comparison table and plots
```

First milestone: the baseline and one alternative norm complete a short train–save–resume–analyze cycle with identical next-batch hashes and correct causal prefixes. Only then schedule the normalization sweep. The main training loops should fail loudly on unknown data revisions, padded-token targets, duplicate run IDs, missing norm buffers, unsupported kernels, or budget exhaustion.

## 12 Red team reviews and final revisions

This section records two distinct review passes and the changes they require. It is part of the implementation specification; open risks remain measured quantities, not reasons to omit the experiments.

### Round one Mathematical and interpretation review

| Challenge | Consequence | Required resolution |
| --- | --- | --- |
| “RMSNorm is just a scalar, so weight graphs are causal graphs” | Misses radial coupling and source-token reweighting | Preserve norm nodes; analytical Jacobian and live/clamped interventions |
| “Lift normalization into one extra coordinate” | May hide the nonpolynomial computation in a dependent coordinate | Specify the lift and its dependency; no automatic global polynomial claim |
| “The global tensor is compact, so every expectation is tractable” | Repeated-input contractions and symmetrization can be expensive | Use local contractions; verify assumptions before global metrics |
| “Norm-free” tanh/erf architectures are tensor models | Replaces one nonpolynomial operation with another | P/N/G operator audit |
| “Group RMS makes the model modular” | Dense matrices still mix groups | Track both content and normalization dependencies |
| “A low-rank denominator gives full RMS stability” | Directions outside its metric can grow unchecked | Nullspace stress tests; disclose any isotropic floor that restores dense dependence |
| “A sparse factor graph is unique” | Gauge transformations change apparent structure | Symmetrization, scale checks, subspace stability, causal tests |
| “Pre/output normalization are independent improvements” | Homogeneous branches can make some combinations redundant | Algebraic equivalence test; distinguish parameterization from expressivity |
| “Batch/running norm is harmless in autoregressive training” | Current-sequence statistics can leak future tokens | Lagged updates outside the forward graph; frozen evaluation and prefix invariance tests |

### Round two Experimental and systems review

| Challenge | Consequence | Required resolution |
| --- | --- | --- |
| “100M tokens reliably picks the winner” | Slow starters and optimizer/horizon effects are discarded | Retain a middling candidate, extend several, report rank changes |
| “One optimizer setting is a fair architecture comparison” | Normalization and optimizer interactions bias results | Equal local LR searches and a finalist optimizer cross-check |
| “Short context establishes long-context quality” | Attention/norm length effects remain hidden | Direct 2K runs and extension branches with matched target positions |
| “500M confirms width transfer” | Simultaneous depth changes confound inference | Add a fixed-depth width check |
| “A fast forward kernel accelerates training” | Backward or optimizer may dominate | Full gradient tests and full-step profiling |
| “Fusing L and R is purely an implementation detail” | Muon on a concatenated matrix changes training | Preserve separate optimizer matrices while fusing only computation |
| “Reference GPU FLOPs predict the bill” | Utilization, compilation, and failed jobs dominate tiny runs | Measured throughput, per-stage limits, global spend ledger |
| “16 bytes per parameter means the model fits” | Attention matrices and logits exhaust VRAM | Peak-memory profiling; bounded logits; fused/checkpointed attention |
| “Cached data and a seed guarantee provenance” | Streaming revisions and packing change examples | Immutable shards plus document/token mapping and next-batch hashes |
| “EMA norm buffers are trivial” | Checkpoint recomputation or accumulation updates them multiple times | One explicit post-step update; save/replay buffers |
| “All successful ablations can be combined” | Interactions erase gains | Test the combined finalist before reporting it |
| “More agents always mean cheaper kernels” | API/search cost exceeds saved GPU time | Bounded attempts and a break-even calculation |

Remaining empirical questions: which norm package wins after 1B+ tokens; whether fixed-scale polynomial models remain stable with depth and shifted inputs; how strongly content overlaps predict useful causal interventions; whether custom kernels improve end-to-end training enough to justify their cost; and whether the small-model rankings persist at 3B and beyond. These become the objectives of the funded follow-up.

### Validation completed for this brief

Tiny NumPy FP64 checks verified the RMSNorm and general-metric Jacobians against centered finite differences, normalized bilinear factorization, symmetrized local tensors, the homogeneous cancellation identity, and the failure of its residual generalization. All five causal-attention input gradients matched finite differences with maximum directional error below `2e-10`; the recurrent and direct implementations agreed to below `1e-14`. The shared pre-RMS attention factor `r_i^-2 r_j^-3` also matched numerically. Model counts, the beta2 conversion, the YAML example, and reference identifiers were checked. These are mathematical/reference checks, not GPU kernel, VRAM, throughput, or training results; those remain Stage A work.

## 13 Primary sources and implementation references

The source summaries above distinguish reported findings from proposed experiments. Kernel versions and pricing should be rechecked at execution time; pin actual commits rather than installing an unrecorded moving branch.

- **R01.** Gonzalez, Smith, Dooms et al. *When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability*. TPM 2026. [Paper](https://tractable-probabilistic-modeling.github.io/tpm2026/papers/w72bKopyg3.pdf). Scope: structured multilinear models and their tensor metrics.
- **R02.** Dooms et al. *Compositionality Unlocks Deep Interpretable Models* (2025). [Paper](https://arxiv.org/abs/2504.02667). χ-nets, ODT, and running-RMS experiments on image classification.
- **R03.** Pearce et al. *Bilinear MLPs enable weight-based mechanistic interpretability* (2024). [Paper](https://arxiv.org/abs/2410.08417).
- **R04.** Zhang and Sennrich. *Root Mean Square Layer Normalization* (2019). [Paper](https://arxiv.org/abs/1910.07467).
- **R05.** Nguyen and Salazar. *Transformers without Tears* (2019). [Paper](https://aclanthology.org/2019.iwslt-1.17/). ScaleNorm/FixNorm evidence is primarily translation.
- **R06.** Bachlechner et al. *ReZero is all you need* (UAI 2021). [Paper](https://proceedings.mlr.press/v161/bachlechner21a.html).
- **R07.** Zhai et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse* (2023). [Paper](https://arxiv.org/abs/2303.06296). sigmaReparam is a weight-only reparameterization.
- **R08.** Wang et al. *SpanNorm* (ICML 2026). [Paper](https://arxiv.org/abs/2601.22580). Use the actual residual topology and initialization.
- **R09.** Wang et al. *DeepNet: Scaling Transformers to 1,000 Layers* (2022). [Paper](https://arxiv.org/abs/2203.00555).
- **R10.** Zhu et al. *Transformers without Normalization* (CVPR 2025). [Paper](https://arxiv.org/abs/2503.10622). DyT uses tanh.
- **R11.** Chen et al. *Stronger Normalization-Free Transformers* (CVPR 2026). [Paper](https://arxiv.org/abs/2512.10938). Derf uses erf; evidence is not a bilinear LLM scaling result.
- **R12.** Loshchilov et al. *nGPT* (ICLR 2025). [Code and paper](https://github.com/NVIDIA/ngpt).
- **R13.** Yang et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (2022). [Paper](https://arxiv.org/abs/2203.03466).
- **R14.** Dey et al. *Don't be lazy: CompleteP enables compute-efficient deep transformers* (NeurIPS 2025). [Paper](https://arxiv.org/abs/2505.01618). [Minimal implementation](https://github.com/EleutherAI/nanoGPT-mup/tree/completep).
- **R15.** Mlodozeniec et al. *Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration*. [Paper](https://arxiv.org/abs/2512.22382). Consult its current version for the exact parameter-group rules.
- **R16.** Bjorck et al. *Scaling Optimal LR Across Token Horizons* (ICLR 2025). [Paper](https://arxiv.org/abs/2409.19913).
- **R17.** Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (NeurIPS 2024). [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8b970e15a89bf5d12542810df8eae8fc-Abstract-Conference.html). [Code](https://github.com/epfml/schedules-and-scaling/).
- **R18.** Marek et al. *Small Batch Size Training for Language Models* (NeurIPS 2025). [Paper](https://arxiv.org/abs/2507.07101).
- **R19.** Porian et al. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models* (NeurIPS 2024). [Paper](https://arxiv.org/abs/2406.19146).
- **R20.** Ai2. [OLMo-core](https://github.com/allenai/OLMo-core), [OLMo 3 training recipes and data manifests](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3).
- **R21.** Wen et al. *Fantastic Pretraining Optimizers and Where to Find Them* (ICLR 2026). [Paper](https://arxiv.org/abs/2509.02046).
- **R22.** Khona et al. *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales* (2026). [Paper](https://arxiv.org/abs/2607.20548). [Code](https://github.com/NVIDIA-NeMo/Emerging-Optimizers).
- **R23.** Li et al. *Hyperband* (JMLR). [Paper](https://www.jmlr.org/papers/v18/16-558.html).
- **R24.** Ai2. *OlmPool: How small architectural choices compound to undermine long context extension* (2026). [Research report](https://allenai.org/blog/olmpool).
- **R25.** Wijmans et al. *Cut Your Losses in Large-Vocabulary Language Models* (ICLR 2025). [Paper](https://arxiv.org/abs/2411.09009). Record gradient filtering if enabled.
- **R26.** Triton. [Fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
- **R27.** PyTorch. [FlexAttention documentation](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html).
- **R28.** Manifest AI. [Retention kernels](https://github.com/m-a-n-i-f-e-s-t/retention), [Scaling Context Requires Rethinking Attention](https://arxiv.org/abs/2507.04239).
- **R29.** NVIDIA. [H100 specifications](https://www.nvidia.com/en-sg/data-center/h100/). Distinguish sparse theoretical peak from dense measured throughput.
- **R30.** NVIDIA. [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
- **R31.** PyTorch. [FSDP principles](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/), [distributed checkpointing](https://pytorch.org/blog/performant-distributed-checkpointing/). Use current APIs from the pinned framework.
- **R32.** Nebius. [GPU pricing](https://nebius.com/prices). H100 public reference $4.50/GPU-hour effective October 2026; availability and instance minimums are separate.
- **R33.** PyTorch. [KernelAgent](https://github.com/meta-pytorch/KernelAgent), [hardware-guided optimization report](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/).
- **R34.** Poddar et al. *KernelOPT* (September 2026). [Paper](https://arxiv.org/abs/2609.30059). Preliminary agent-optimization evidence; require training-specific validation.
- **R35.** Hugging Face. [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

