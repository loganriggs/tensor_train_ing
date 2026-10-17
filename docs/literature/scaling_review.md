# Scaling review: hyperparameter transfer, batch, schedules, stability, optimizers, systems, methodology

Literature review written for scaling the bilinear-MLP / signed-product polynomial-attention model from ~100M–500M
(measured on one RTX 5000 Ada, `docs/scaling.md`) toward dense 1B–32B. Date: 2026-10-16.

**How to read the evidence labels.** **[E]** means established: the cited paper reports it directly and I checked
the claim against its abstract or full text. **[S]** means speculative: my own inference for our architecture, not
shown by any paper. **[U]** means I could not verify the claim from the paper itself in this review (written from
memory, from a secondary summary, or from a garbled extraction). Check [U] items before relying on them.

**One code fact that changes the reading of the literature.** `tnlm/optim.py` already scales the hidden-matrix
weight decay *inversely* to the LR multiplier (`weight_decay_hidden / width_mult`, `lr * width_mult`). The product
η·λ is therefore width-invariant. Our empirical rule is η ∝ w^−½ with λ ∝ w^+½. That is the setting
Kosson et al. (2025) single out as a likely-better, untested alternative to μP (§1.3). Other current defaults: the
PyTorch Adam ε (1e-8, not set in config), β2 = 0.98726 at 8,192 tokens per update, and the same β2 in
`funded_32b.yaml` at 2.1M tokens per update.

---

## Executive summary: the 10 most actionable recommendations

1. **Keep the ½-exponent width rule, and treat it as theory-consistent rather than an anomaly.** Our η ∝ w^−½
   (with η·λ fixed) is what the "no-alignment" analysis of Everett et al. predicts for Adam hidden layers
   [Everett24]. Kosson et al. also find that when tokens per batch ≫ width, update alignment decays roughly as
   width^−½ [Kosson25]. Ours is 8,192 tokens against widths of 256–2,048, so we are in that regime. Next: run the
   planned **4× width test (512→2048)**, adding one arm with λ held fixed (μP-style). That checks whether the
   λ ∝ w^½ half of the rule matters at 50M+ tokens. *Ties to: coordinate-check slopes +0.07–0.12 under the ½ rule;
   textbook μP over-corrects (−0.45).*
2. **Re-run the coordinate check later in training, not only at step 20, and log per-layer alignment.** μP's
   assumptions hold mainly early in training. Later, the weight-decay equilibrium sets update size, and relative
   updates scale as √(ηλ) [Kosson25; 2510.15262]. Measure ‖ΔW‖/‖W‖ and the alignment ratio per layer at roughly
   1k and 5k steps, at two widths. Separately, sweep the **embedding LR** (×{1, 3, 10}). Recent work attributes
   most of μP's advantage over SP to the embedding LR alone [Kalra26]. *Ties to: confirmation at 50M tokens covered
   only two LRs.*
3. **Adopt full CompleteP (α = 1) for depth, including its ε and norm-gain rules, and test a ≥4× depth ratio
   before 64 layers.** CompleteP α = 1 beat α = ½ and μP at 1.5B: 12–34% FLOP savings, and the largest gain at
   the deepest shape (179 layers) [CompleteP]. Complete^dP reports transfer for any α ∈ [½, 1] [CdP]. The 32B
   preset goes from 12 to 64 layers (5.3×). Run 12→48 layers at width 512 on 50M tokens, 2 seeds. *Ties to: a
   (12/L) branch multiplier flattened per-layer updates and won at 24 layers in the pilot only.*
4. **Measure the token-horizon LR exponent ourselves before launching any ≥1B run.** The optimal peak LR falls
   with training tokens D: about D^−0.32 for 350M–2.7B GPT-3 models under cosine [Bjorck25], and about
   1/√(step multiplier) in Complete^dP [CdP]. At width 512, find the LR optimum at two horizons (for example
   50M and 400M tokens) on a finer grid (√2 spacing). Our 3× grid cannot resolve shifts of this size. *Ties to:
   base LR 4.5e-4 was chosen at 4.8M–50M tokens and is used unchanged for 32B.*
5. **Set batch size from a measured critical batch size (CBS), not from the 8,192-token default. Scale LR by
   √(batch multiplier) and keep β2 sane.** CBS grows with **data**, not model size [Zhang25CBS; PowerLines]. It is
   about 0.4–1.3M tokens at Chinchilla data for 85M–1.2B [Zhang25CBS], and it plateaus near 4k×4k-token sequences
   (~17M tokens) late in OLMo 1B/7B runs [Merrill25]. Measure CBS with branched runs from a pilot checkpoint
   [Merrill25]. Use the Adam √-rule for LR [CdP; Merrill25], and use a batch-size ramp. The β2 token-half-life rule
   [Marek25] is for small batches: blindly extrapolated to 2M tokens it gives β2 ≈ 0.04 from 0.95 at 32k. Keep β2
   in 0.95–0.99 at large batch and tune it. *Ties to: `global_prediction_tokens_per_update` and β2 are flagged
   unvalidated in every scaled config.*
6. **Schedule: warmup → constant → decay toward zero, with ~20% cooldown at short rungs.** WSD-style cooldown
   matches cosine at about 20% of steps, and the 1−√ shape beats linear [Hagele24]. The required fraction falls
   for long runs. Linear decay **to zero** beats 10×-cosine by a large margin at high tokens-per-parameter
   [Bergsma25D2Z]. Our 10% linear cooldown is fine for ranking. For final quality, branch a 20% 1−√ cooldown to
   (near) zero. Do not adopt schedule-free as the default: it is worse at large batch [Zhang25CBS].
7. **Instrument and guard the degree-5 attention before scaling. Prefer guards that keep the polynomial
   form.** Attention-logit growth and output-logit divergence appear in small models at high LR and predict
   large-scale divergence [Wortsman23]. Muon makes attention-logit blowups more likely [KimiK2]. Log per layer: max
   and RMS of S1, S2 and S1·S2; Q/K/V/O/L/R/D weight norms; max and kurtosis of the D input (Lx)⊙(Rx); log Z of the
   output; and the fraction of Adam denominators near ε. If scores grow, the first guard is **QK-clip** (rescale
   Q/K weights after the step; the forward stays polynomial) [KimiK2]. **QK-norm** is the second test. It adds
   normalizer nodes, and it is the largest single factor in OlmPool's long-context degradation [OlmPool]. Add
   z-loss 1e-4 as a shared ablation [Wortsman23; OLMo2]. *Ties to: trained models produce non-finite logits under
   constant denominators, so scale control is load-bearing.*
8. **Keep BF16 with FP32 accumulation of S1·S2 and of (Lx)⊙(Rx). No FP8 until outliers are measured. Lower Adam ε
   at ≥7B.** SwiGLU's FP8 failure at about 200B tokens comes from the gate weights aligning until each neuron is
   **quadratic** [Fishman24]. A bilinear MLP is quadratic by construction, so outlier amplification is our default
   regime [S]. ε = 1e-8 is fine up to ~1B in SP. Use 1e-12 to 1e-15, per-layer ε, or Adam-atan2 for larger models
   [Everett24; Wortsman23].
9. **For Muon, replace `match_rms_adamw` with spectral (√(d_out/d_in)) scaling before any width-transfer claim,
   and budget a 1.1–1.4× gain, not 2×.** Our Muon coordinate-check slopes of +0.4–0.7 match the finding that the
   Kimi-style RMS-matching rule does not transfer across width, while √(d_out/d_in) does [Qiu25]. The measured
   advantage over well-tuned AdamW shrinks from ~1.4× at 0.1B to ~1.1× at 1.2B [Wen25], or holds ~1.4× to 1.4B
   with correct scaling [Qiu25]. Matrix optimizers do hold up better at very large batch [R22; EssentialMuon].
   That matters at 32B, where batches are large.
10. **Systems and methodology for the next stage.** (a) At ≥7B use FSDP2 with full per-block checkpointing. By
    our 66·d bytes/token/layer figure, 32B uncheckpointed activations are ~212 GB/GPU at 8k tokens. Use
    every-other-layer selective checkpointing at 1–3B [TorchTitan]. Expect 33–42% MFU for well-tuned dense
    training on H100 [TorchTitan]. Re-measure the 0.74× poly/softmax throughput ratio at d = 2048–5120, where
    attention is a smaller share of FLOPs [S]. Keep intra-document masking [Zhao24]. (b) For extrapolation, use ≥5
    model sizes, fit on intermediate checkpoints after dropping the early part, and run ≥3 seeds whenever the
    effect is below ~0.03 CE [Hitchhiker; Porian24]. Set warmup tokens ≈ N and count unembedding FLOPs [Porian24].
    Add an early context-extension probe [OlmPool]. *Ties to: same-config seed spread of 0.01–0.06 CE at 50M tokens
    already exceeds some effects we report.*

---

## 1. Hyperparameter transfer across width, depth, batch and duration

### 1.1 μP / Tensor Programs V [R13]
**Shows:** zero-shot transfer of LR, init and multipliers across width under the maximal-update parameterization.
For Adam, μP scales hidden-matrix LR as 1/width and the output multiplier as 1/width. **Relevance:** μP's 1/width
rests on a *full alignment* assumption: updates ΔW correlate with the incoming activations, so ‖ΔW x‖ grows by a
factor of width. **Caveat for us:** we hold head dimension fixed and add heads, so μP's 1/d_head attention scaling
never comes into play. Our width question is purely about matrix LRs and the readout. [E]

### 1.2 Everett et al., "Scaling Exponents Across Parameterizations and Optimizers" (ICML 2024)
**Shows [E]:** tens of thousands of models, up to 26.8B parameters, 3 optimizers × 4 parameterizations. Under
*full alignment*, SP+Adam hidden and readout LRs scale as O(1/n). Under *no alignment* the exponents are about ½
shallower, i.e. ~1/√n for hidden layers. Measured alignment starts at about 0.5 and moves during training: readout
alignment peaks at 0.7–0.8 under SP, and MLP alignment stays at or below 0.8. So reality sits between the two
assumptions. With per-layer constants tuned, *all* parameterizations transfer. SP with per-layer LRs beat μP: their
15.3B SP model outperformed the 26.8B μP model. Adam ε underflows as gradients shrink with width. Default 1e-8 is
"likely acceptable" for SP up to ~1B; use 1e-12 to 1e-15, per-layer ε (base 1e-12), or **Adam-atan2**
(update = atan2(m, √v), with no ε). 1e-30 hurt at 26.8B.
**Relevance:** this gives a direct theoretical reading of our 0.5 exponent. The coordinate check measures the
*no-alignment* regime: updates whose overlap with activations grows only as √width. A global SP LR fitted
empirically lands between the two predictions. **[U]** I could not read the empirical global-LR SP exponent cells in
their tables (garbled extraction), so I do not claim they measured 0.5.

### 1.3 Kosson et al., "Weight Decay may matter more than μP for LR Transfer in Practice" (ICLR 2026)
**Shows [E]:** LLaMA-style models up to 1B (width 2048, up to 16× width ratio), about 20B DCLM tokens, ~1M-token
batches. μP's alignment assumptions hold only at the start of training. When batch B ≫ width C, update alignment
becomes width-dependent, falling to about C^−½, and the alignment ratio drops to ~1 at every width. Later in
training, relative update size is set by the AdamW equilibrium ∝ √(ηλ), not by η alone. μP transfers in practice
only with **independent weight decay** (η → η/m, λ → mλ, so ηλ is constant). Otherwise μP mainly acts as an
implicit warmup. The authors suggest (η/√m, √m·λ) as possibly better, but did not test it.
**Relevance:** (i) Our coordinate check runs at 8,192 tokens/step against widths of 256–2048, which is the B ≫ C
regime where they find ~C^−½ update alignment. That is a direct mechanism for our 0.5 exponent, and it explains
why 1/width over-corrected. (ii) Our code already implements the (η/√m, √m·λ) variant. Our 50M-token result
(η ∝ w^−½ plus λ ∝ w^½ beats the default at both LRs and both seeds) is thus a small-scale positive data point for
their untested suggestion [S]. **Caveat:** their models use softmax/SwiGLU, and the equilibrium argument assumes
roughly scale-invariant weights. Ours are not scale-invariant (§4.6).

### 1.4 Related 2025–2026 work on why μP does or does not transfer
- **Haas et al. (NeurIPS 2025, 2505.22491).** [E] Under SP, the optimal LR shrinks *more slowly* than the
  infinite-width O(1/n) stability bound. Cross-entropy permits a "controlled divergence" regime: logits blow up but
  activations and gradients stay stable, and features still learn. **Relevance:** this explains why our
  *logit* slope (+0.90 under default) needed its own ½ multiplier without the run diverging. The abstract gives no
  specific exponent [U for "½"].
- **"Robust Layerwise Scaling Rules by Proper Weight Decay Tuning" (2510.15262).** [E] Training reaches an
  optimizer-governed steady state. The top singular value scales empirically as √(η/λ)·d^0.75, which motivates
  λ ∝ √d paired with μP's η ∝ 1/d. That gives η·λ ∝ d^−½. Ours gives η·λ ∝ d^0. Both papers agree that
  λ should grow with width [S on which product is right for us; recommendation 1 tests it].
- **Kalra & Barkeshli (ICML 2026, 2605.21486).** [E] Most of μP's advantage over SP+AdamW comes from the larger
  **embedding LR**. In SP the embedding LR is a bottleneck that causes instability. Weight decay improves
  scaling-law fits but hurts extrapolation at fixed tokens-per-parameter. **Relevance:** our embeddings (init std 1)
  use the unscaled base LR, so the embedding-to-hidden LR ratio grows as √(d/512). This has not been tuned.
- **Qiu et al. 2512.05620; "μP for more optimizers" 2602.20937.** μP-style scaling for Muon/Shampoo/SOAP (§5).

### 1.5 u-μP (Blake et al. 2024)
**Shows [E]:** combines μP with Unit Scaling so activations, weights and gradients start at unit scale. Defaults
are near-optimal, and it works out of the box in FP8. **Relevance:** attractive if we later go to FP8. But unit
scaling assumes known nonlinearity moments, and our degree-5 attention breaks the usual moment bookkeeping [S].
Model scales are not in the abstract [U].

### 1.6 Depth: Depth-μP, CompleteP, Complete^dP
- **Depth-μP / Tensor Programs VI (Yang et al. 2023).** [E] Gives a unique optimal parameterization for
  one-layer-per-block ResNets. It reports *fundamental limitations* for blocks of depth ≥ 2, which includes
  Transformer and bilinear blocks. The 1/√L multiplier is the depth-μP choice [U from abstract; Bordelon et al.
  2309.16620 make a similar argument].
- **CompleteP (Dey et al., NeurIPS 2025) [R14].** [E] α = 1: residual branch × L_base/L; hidden init variance
  and Adam LR scale with width only (no depth factor); norm-gain/bias LR unchanged; hidden weight decay × m_N; Adam
  ε × 1/(m_N·m_L) in residual blocks (a later version corrected the ε rules, and the authors note a "small enough"
  ε suffices). Transfer was shown from 2 to 128 layers on 300M tokens. At 1.5B and 20 TPP: 11.8% FLOP saving at the
  optimal shape and 34.4% at the deepest shape (L = 179) versus μP. α = ½ was *worse* than μP (−19.6% and −23.7%).
  **Relevance:** our (12/L) multiplier is CompleteP α = 1. Our finding that a global LR depth factor starves early
  layers (first-block updates ∝ L^−0.6) matches their "non-lazy in all layers" argument.
- **Complete^dP (Mlodozeniec et al. 2512.22382) [R15].** [E] Extends CompleteP to batch and duration, with
  per-module HPs (13 module types, 79 HPs). Rules: hidden LR × m_N^−1 × m_L^(α−1); residual multiplier m_L^−α with
  α ∈ [½, 1] all transferring; hidden ε × m_N^−1 m_L^−α; λ × m_N. For batch, the SDE **√-rule** (η, λ × √κ;
  steps × 1/κ) is "critical to good LR transfer". For duration, η_opt ∝ 1/√κ in steps under cosine. Tuned at
  50M/1.6B tokens; transfer to 1.3B/26B tokens gives a 27% speed-up, and 7.2B/143B runs a 1.32× speed-up
  (appendix). **[U]** The β-scaling and ε entries of their Table 1 were partly garbled in my extraction, so read
  the paper before implementing β rules. There is also an internal inconsistency on the compute ratio (420× vs
  ~600×).

### 1.7 What theory predicts for bilinear and polynomial layers specifically [S]
No paper derives μP or CompleteP for a no-activation bilinear MLP or for softmax-free product attention. These
first-order arguments are mine:
- **The width exponent is unchanged to first order.** For h = (Lx)⊙(Rx), Δh = (ΔL x)⊙(Rx) + (Lx)⊙(ΔR x) + O(Δ²).
  Each first-order term is a linear-layer update times an O(1) partner, so the per-matrix width rule matches a
  linear layer. Tensor Programs admit coordinatewise multi-argument nonlinearities such as a·b, and μP is routinely
  used with GLU MLPs [Everett24; CompleteP use gated MLPs; U on exact architecture]. So our ½ exponent should apply
  to L, R and D, and our coordinate check found MLP slopes close to attention slopes.
- **Multiplicative depth raises the constant and the sensitivity to scale drift, not the exponent.** Relative
  changes add across factors: δ(MLP) ≈ δL + δR + δD, and δ(attn) ≈ δQ1 + δK1 + δQ2 + δK2 + δV + δO. So the polynomial
  branch is about 2–3× more sensitive per step than a linear branch. Equivalently, its effective LR is larger by a
  constant, which matches the steep high-LR penalties we see. Weight-norm growth enters as ‖W‖³ (MLP) and ‖W‖⁶
  (attention including O). Unlike softmax, nothing saturates. So the LR × WD equilibrium (√(η/λ) weight norms
  [2510.15262]) sets the *function's scale*, not only step size. This is why recommendation 7 logs per-matrix
  norms.
- **No softmax means no per-row normalization.** Output scale depends on n_visible and on document length. The
  brief's n^−½ factor handles the mean case only. Pre-RMSNorm controls input scale but not the growth of
  ‖Q1‖‖K1‖‖Q2‖‖K2‖.

---

## 2. Batch size, β2, ε and token horizon

### 2.1 Critical batch size
- **Zhang et al., "How Does Critical Batch Size Scale in Pre-training?" (ICLR 2025, 2410.21676).** [E]
  85M–1.2B on C4, context 512. CBS scales **primarily with data**: at fixed data the model-size exponent is ~0.09;
  at Chinchilla-optimal data, CBS rises from ~740 sequences (~0.38M tokens) at 85M to ~2,550 sequences (~1.3M
  tokens) at 1.2B. Hyperparameters: β2 = 0.95 suits large-batch short runs, and 0.99–0.999 helps small batches
  (<~262k tokens) and long runs. β1 = 0.95 was slightly better than 0.9. **Constant LR plus EWA matches cosine and
  beats WSD**, especially at large batch. Schedule-free is competitive at small batch but clearly worse above 1,024
  sequences. Context 2^9–2^12 gives similar batch scaling. **Caveat:** their printed data-size formula does not
  reproduce their own table; use the table values.
- **Bergsma et al., "Power Lines" (2505.13738).** [E] B_opt and B_crit are power laws in D, independent of N. The
  AdamW timescale τ = B/(ηλD) has an optimum that follows a power law in tokens-per-parameter, and λ_opt is linear in
  B at fixed N and D. Run on Cerebras.
- **Merrill et al., "Critical Batch Size Revisited" (2505.23971).** [E] Measures CBS directly by branching
  from checkpoints with batch k·B and LR √k·η, over Δ = 2B tokens, with loss tolerance 0.01 (arbitrary). CBS
  starts near 0, grows fast, then plateaus at about 4,096 sequences of 4,096 tokens for OLMo 1B and 7B alike.
  **Batch-size warmup** (double the batch when CBS > 2× batch; LR × √2) reached slightly better loss with 43%
  fewer steps on OLMo 1B. **Relevance:** cheap to reproduce on our 300M and 1B pilots. It also directly gives the
  batch schedule for 7B/32B.

### 2.2 Small batches and β2 [R18]
**Marek et al. (NeurIPS 2025, 2507.07101).** [E] Batch size 1 trains stably and is more robust to HPs. With β2
scaled to keep the **token half-life** fixed (β2* = β2^(B*/B)), small batches match or beat large ones per FLOP.
Vanilla SGD at batch 1 matches AdamW on GPT-3 1.3B. The √-LR rule fails at small batch: going from 1 to 1,024,
about 3× works, not 32×. They recommend the smallest batch that maximizes throughput and no gradient accumulation
except across replicas. **Relevance and caveat [S]:** our β2 = 0.98726 at 8,192 tokens follows this rule. The rule
is a timescale identity tested toward *smaller* batches. Extrapolated to 2.1M-token batches it gives an unusable
β2. At large batch, follow Zhang et al. and Porian et al.: β2 ∈ [0.95, 0.99], tuned. The current
`funded_32b.yaml` value (0.987 at 2.1M) sits inside that range by accident.

### 2.3 Token-horizon LR [R16]
**Bjorck et al. (ICLR 2025, 2409.19913).** [E] LR*(D) = B·D^−β, with β ≈ 0.32–0.42 for 350M–2.7B and ≈ 0.65–0.70
for 50M–125M. Joint fit: LR*(N, D) = C·N^−0.23·D^−0.32. μP does not remove the horizon dependence (50M check).
Under cosine to 0.1×, LLaMA-1-7B's 3e-4 was more than 2.5× too high. **Relevance:** small models give inflated
β. So a horizon exponent measured at width 512 (~100M) will *over*-estimate the shrink needed at 1B+. Fit it at
the 300M–500M rung if possible. **Caveat:** horizon effects may be weaker under WSD or EWA than under cosine.
Complete^dP notes the same schedule dependence.

### 2.4 Other batch and LR laws
- **Porian et al. [R19].** [E] β2 = 0.95 was suboptimal at batch ≤128 sequences; 0.99 or 0.999 fixed it. Tuned
  LR and batch follow power laws in N (e.g. LR ≈ 2.4e-3 at batch 640 for 901M). Their batch-size exponents agree
  with DeepSeek's within 0.05, but LR exponents differ by 0.11, i.e. a 2–3× disagreement in predicted LR.
- **DeepSeek LLM (2401.02954) and "Predictable Scale / Step Law" (2503.04715)** fit B_opt and η_opt as power laws
  in compute. **[U]**: I did not re-check their exponents. Porian's comparison above suggests treating such laws as
  ±2–3× in LR.

### 2.5 ε
See §1.2 and §4.1. Practical: track the fraction of entries with √v < 10·ε per parameter group. In OLMo 2,
lowering ε from 1e-5 to 1e-8 stabilized early gradient norms [OLMo2, E]. Wortsman et al. improved a 4.8B model by
lowering ε to 1e-15 [E].

---

## 3. Schedules: WSD, decay-to-zero, schedule-free, cooldown length

- **Hägele et al. (NeurIPS 2024) [R17].** [E] Constant LR plus cooldown matches cosine. Benefits plateau at a
  cooldown of about 20% of steps. A **1−√ cooldown** consistently beat linear, with a larger advantage on longer
  runs; with 1−√, 5% nearly matched cosine at 20B tokens. **The required fraction shrinks as training grows:** in a
  200k-step run, 10k cooldown steps nearly matched cosine. Windowed weight averaging (SWA) helps but does not close
  the gap. Schedule-free was sensitive to its momenta and did not beat cooldown. Scales: 33M–360M (up to 10B
  tokens), 1B at 100B/460B tokens, and a short 8B run.
- **Bergsma et al., "Straight to Zero" (ICLR 2025, 2502.15938).** [E] Linear decay *to zero* beats 10×-decay
  cosine at compute-optimal and larger data. A 610M model at 80 TPP with decay-to-zero beat 200 TPP with 10× decay,
  about 60% compute saved. They interpret AdamW as an EMA of updates: decaying late averages over more updates.
- **Song et al., "Through the River" (NeurIPS 2025, 2507.09846).** [E] At 124M and 6B tokens, schedule-free AdamW
  follows the "river" with no decay phase or averaging. Small scale only. **ScheduleFree+ (2605.19095)** claims
  large-scale wins over WSD (including +31% at 1000 TPP). **[U]**: I saw only the abstract and a third-party summary.
- **Zhang et al. (CBS paper).** [E] Constant LR plus EWA ≥ WSD at large batch. Schedule-free is worse above
  1,024 sequences.

**For us.** Branched cooldowns are the right protocol for ranking (cooled vs cooled). For final models, use a
1−√ or linear decay to ~0 over 10–20% of steps; more for short rungs, and a fixed absolute length for long runs is
defensible. An EWA of weights (decay 0.998–0.9995 per step at our step counts [Zhang25CBS]) is a cheap way to
evaluate "cooled-like" checkpoints without branching. Validate it once against a real cooldown on our model [S].
Interaction to watch [S]: cooldown changes the AdamW equilibrium (√(η/λ) norms). For a degree-5 function, the
weight-norm shift during cooldown changes output scale more than in softmax models. Monitor logits during
cooldown branches.

---

## 4. Stability at scale for multiplicative, polynomial and normalization-light models

### 4.1 General Transformer instabilities
- **Wortsman et al., "Small-scale proxies…" (ICLR 2024, 2309.14322).** [E] Covers 2.4M–4.8B. Attention-logit
  growth (driven by q/k *norms*, not their alignment) is fixed by **qk-layernorm**, which let a 1.2B model train at
  LR 0.3. Late output-logit divergence without weight decay is fixed by **z-loss 1e-4**. Longer warmup and
  **independent (LR-decoupled) weight decay** reduce LR sensitivity. μP stabilizes the optimal LR but not LR
  sensitivity, and does not replace qk-layernorm. Depth raises LR sensitivity faster than width. Max-attention-logit
  vs size predicted a 4.8B divergence that then happened; logits above ~1e4 coincided with divergence. ε near
  gradient RMS shrinks updates; 1e-15 helped. **Relevance:** their protocol, sweeping LR over orders of magnitude
  at small scale and tracking scaling of activation and gradient norms, is exactly how to stress-test our
  attention before 1B. Our 3-point LR grid is too narrow for that purpose.
- **OLMo 2 (2501.00656).** [E] Stability recipe: QK-norm (RMSNorm on q and k), z-loss (1e-5 in Table 1 vs 1e-4 in
  §3.3.3, inconsistent), **reordered norm** (norm on block *outputs*), no weight decay on embeddings, ε 1e-8, and
  N(0, 0.02) init instead of depth-scaled init. Diagnostics: a spike score (fraction of values ≥7σ from a rolling
  mean of loss and gradient norm), a growth exponent across layers at init, and activation and gradient norms vs
  √d_model.
- **σReparam [R07].** A weight-only spectral reparameterization against attention entropy collapse. For a
  softmax-free model "entropy" is undefined, but the weight-only nature preserves the polynomial form at each
  checkpoint [S].
- **Kimi K2 / MuonClip (2507.20534).** [E via blog and report excerpts] Muon at scale showed exploding attention
  logits more often than AdamW. Soft-capping and QK-norm were judged inadequate (in part because their MLA design
  complicates QK-norm [U]). **QK-clip** rescales W_q and W_k after each step when the max logit exceeds a
  threshold (100 reported by a secondary source [U]). Trained on 15.5T tokens with no loss spikes. **Relevance:**
  QK-clip changes only weights, so the trained model remains exactly a polynomial network. That suits our
  interpretability goal better than QK-norm [S]. The analogue for us is clipping max |S1|·|S2| or max |S1·S2| per
  head, splitting the rescale across Q1, K1, Q2 and K2.

### 4.2 Outlier features and precision
- **Fishman et al., "Scaling FP8 training to trillion-token LLMs" (ICLR 2025, 2409.12517).** [E] 7B models on 2T
  tokens. FP8 diverged after about 200B tokens because SwiGLU neurons' w1 and w2 **align** (driven by ℓ2
  regularization, even weak), making each neuron effectively **quadratic** and producing extreme outliers. BF16 did
  not diverge. **Smooth-SwiGLU** (per-channel scale on the linear branch, inverse scale on the down projection) fixed
  FP8 and also gave smoother BF16 training at higher LR. **Relevance [S]:** our MLP is quadratic everywhere from
  step 0, so the outlier mechanism needs no alignment phase. Expect heavy-tailed (Lx)⊙(Rx). Track per-channel max
  and kurtosis of D's input over long runs. If we move to FP8, apply Smooth-SwiGLU-style per-channel scaling, which
  preserves the function exactly and folds into D.
- **Outlier-feature / massive-activation literature** (e.g. He et al. 2405.19279; Sun et al. 2402.17762). Kurtosis
  metrics and norm placement affect outliers [U: not re-checked in this review].
- **BF16 [S].** Products of four BF16 scores lose about 4× the per-factor relative precision. Keep S1·S2 and its
  accumulation into V in FP32 inside the kernel. Our kernel test (BF16 rel < 3e-2 vs FP64) should be re-run at
  T = 4096 with trained-weight score statistics, not only random weights.

### 4.3 Polynomial and linear attention trained at scale
- **Power attention (Manifest, 2507.04239) [R28].** [E] (QᵢᵀKⱼ)^p with **row-sum normalization**, so only even
  p. Uses per-head **gating**. Tested at 124M–400M GPT-2 and RWKV-based long-context models (up to 65k tokens,
  ~1000 H100-hours). The authors expect softmax to overtake it given enough FLOPs on natural language. **Relevance:**
  the closest relative to our attention at scale, but it *does* normalize and gate. Ours does neither.
- **PolySketchFormer (2310.01655).** [E] Degree-p attention (p even) with **layer-normed q and k** and row
  normalization (+1 in the denominator for safety). No quality loss vs softmax up to ~700M. The paper cites prior
  work where **negative attention weights caused instability**. That bears directly on our *signed* product.
- **Based (2402.18668).** [E] Linear attention plus sliding window up to 1.3B, matching Mamba on perplexity. Its
  Taylor feature map is a 2nd-order polynomial [U from abstract; known from the paper body].
- **"The Devil in Linear Transformer" / TransNormer (2210.10340).** [E] Unbounded gradients in kernel linear
  attention hurt convergence. The fix replaces scaling with an **output normalization**. RetNet, GLA, Mamba-2 and
  MiniMax-01 lightning attention (456B MoE, 45.9B active, 1M context [E]) all apply a per-head output norm or
  GroupNorm after unnormalized mixing [U for each specific norm placement]. **Relevance [S]:** every large-scale
  softmax-free mixer I found uses at least one of: row normalization, output normalization, gating, or q/k
  normalization. Ours uses only pre-RMSNorm and a fixed n^−½ factor. That is the riskiest unvalidated aspect of
  going to 32B. A per-head output RMSNorm is the cheapest candidate if needed. As the brief notes, it cancels the
  n_visible factor and adds a normalizer node per head per position.
- **Scaling laws for linear-complexity LMs (Shen et al., EMNLP 2024, 2406.16690).** [E] TNL, HGRN2 and cosFormer2
  from 70M to 7B on 300B tokens scale similarly to LLaMA. This is evidence that non-softmax mixers *can* follow
  ordinary scaling laws, though all of those use normalization or decay.

### 4.4 Bilinear and polynomial MLPs
- **Shazeer, GLU variants (2002.05202).** [E] T5-base (d_ff 2048 for GLU variants), held-out log-perplexity at
  65k / 524k steps: Bilinear 1.960 / 1.648; SwiGLU 1.944 / 1.636; GEGLU 1.942 / 1.633; ReLU FFN 1.997 / 1.677. So
  bilinear sits about 0.012–0.016 behind SwiGLU and far ahead of ReLU FFN.
- **Pearce et al. [R03].** [E] At ~30M TinyStories, Bilinear 1.337 vs SwiGLU 1.321 at equal epochs; essentially
  equal at equal wall time. Converting TinyLlama-1.1B to bilinear cost ~0.05 loss, shrinking with ~400M tokens of
  training. No bilinear-vs-SwiGLU scaling-law study exists [E: none found].
- **PolyCom (ICLR 2025, 2411.03884).** [E] Polynomial activations: PolyReLU, and PolyNorm Σ aᵢ xⁱ/‖xⁱ‖, which
  **normalizes each power**. Tested at 1.3B dense (250B tokens) and an OLMoE-1B-7B MoE, slightly better than
  SwiGLU. The authors credit stability to normalization plus gradient clipping, and built PolyNorm specifically to
  keep powers in BF16/FP16 range. **Relevance:** the largest-scale polynomial-activation evidence, and it also
  needed per-power normalization.
- **Π-nets (Chrysos et al.)** show polynomial nets without activations train in vision. They are not
  LM-at-scale evidence [U: not re-checked].

### 4.5 What this implies for us [S]
Bilinear MLPs are well supported up to ~1B in quality terms. Softmax-free *signed* degree-4 attention without
row, output or q/k normalization has **no** published large-scale precedent I could find. Treat 1B as the first
real stability test. Have QK-clip, z-loss and an output-RMS fallback implemented and tested at 100M **before**
renting H100s. Our finding that the model cannot run on frozen denominators says pre-RMSNorm is doing essential
scale control, and nothing else in the attention path is.

---

## 5. Optimizers at scale

- **Muon is Scalable (Moonshot, 2502.16982).** [E] Two needed changes: weight decay, and per-parameter update-RMS
  matching to AdamW. About 2× compute efficiency vs AdamW in their scaling-law fit. Moonlight is a 3B/16B MoE on
  5.7T tokens, with a memory-optimal distributed implementation.
- **Fantastic Pretraining Optimizers (Wen et al., ICLR 2026) [R21].** [E] With fair per-optimizer tuning and
  end-of-training comparison, matrix optimizers (Muon, SOAP) are fastest, but the speedup falls from ~1.4× at 0.1B
  to ~1.1× at 1.2B (1–8× Chinchilla data). Rankings can flip mid-training because of LR decay.
- **Qiu et al., "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers"
  (2512.05620).** [E abstract] With correct μP-style scaling, Muon, SOAP and Shampoo give ~1.4× over AdamW at
  190M–1.4B. Under incorrect scaling "the speedup vanishes rapidly with scale". Independent λ ∝ 1/width is nearly
  optimal. [U from search summary] Only η ∝ √(d_out/d_in) transferred among the tested Muon rules; the Kimi
  RMS-match rule did not. **Relevance:** our Muon coordinate check (`match_rms_adamw`, slopes +0.4–0.7) is the
  predicted failure. `make_optimizer` currently forbids width scaling with Muon; add a spectral rule instead.
- **Practical efficiency of Muon (Essential AI, 2505.02222).** [E] Muon expands the compute-time Pareto frontier
  and keeps data efficiency well past the CBS. It works with μP (with a "telescoping" sweep), up to 4B.
- **SOAP, Muon and Beyond (Khona et al. 2026) [R22].** [E abstract] Multi-billion-parameter models on trillions
  of tokens. SOAP and Muon consistently beat AdamW. At batches up to 100M tokens they stay stable while AdamW
  degrades. Modified SOAP removes loss spikes. Uses update-RMS matching for cross-optimizer LR transfer.
- **Kimi K2.** Muon increases the risk of attention-logit explosion (§4.1).

**What changes past 1B:** (i) AdamW's advantage gap closes or reverses at large batch, so Muon's value at 32B is
mostly **larger usable batch**, i.e. fewer, more parallel steps. (ii) Attention-logit control becomes necessary
with Muon. (iii) Distributed orthogonalization under FSDP needs gathering full matrices or block-wise variants;
blocking also reduced width drift in Qiu et al. **For us:** keep AdamW as the scaled reference. Muon becomes worth
the engineering at 7B/32B only after spectral width scaling passes a 512→2048 coordinate check and a fair
two-width LR sweep [S].

---

## 6. Systems for 1B–32B dense training with a custom attention kernel

- **TorchTitan (ICLR 2025, 2410.06511).** [E] H100 per-GPU throughput, Llama 3.1 8B with FSDP2 and selective AC:
  6,258 tok/s, 6,674 with torch.compile, 9,409 with Float8 (8 GPUs). 128 GPUs: 5,645 → 9,319. 70B (FSDP 32 × TP 8,
  full AC): about 1,010 tok/s/GPU with AsyncTP. MFU without Float8 is about **33–42%** (8B). Their H100s were
  non-standard (HBM2e, lower TDP). AC options: full, **op-level SAC** (save matmul/SDPA outputs, every other
  matmul), and **layer-level SAC** (every Nth block). The 70B and 405B runs used full AC.
- **Llama 3** reports 38–43% BF16 MFU at 8k–16k H100s [U: not re-checked].
- **FSDP2 [R31].** Per-parameter DTensor sharding. HSDP (shard within a node, replicate across nodes) cuts
  inter-node traffic. For 32B on 16–64 H100s, FSDP2 alone is standard. TP becomes useful when per-GPU microbatches
  get too small to hide all-gathers [S].
- **Activation memory [S, from our numbers].** 66·d bytes/token/layer uncheckpointed gives ~4.6 GB at 1.4B
  (24 × 2048 × 8k tokens), ~70 GB at 7B (32 × 4096 × 8k) and ~212 GB at 32B (64 × 5120 × 8k). So: full per-block
  AC at ≥7B (matching the planner), and layer-level SAC (every 2nd block) at 1–3B. An op-level SAC policy for our
  kernel should **save the poly-attention output (there is no softmax (m, l) statistic to keep) and recompute the
  cheap elementwise products**. The two-kernel backward already recomputes S1 and S2.
- **Attention share at scale [S].** Per token per layer, parameter FLOPs ≈ 2·(6d² + 3d·m) ≈ 32d², while our
  attention costs ≈ 6·T_vis·d (S1, S2, AV), i.e. ≈ 3Td causal. The ratio ≈ 3T/(32d) is ~9% at T = d = 512 and ~7%
  at T = 4096, d = 5120. Measured attention is ~26% of step time at 100M because of kernel efficiency (11 tile
  matmuls in backward), not FLOPs. **Expect the 0.74× ratio to improve at 7B/32B** if the kernel reaches a
  similar fraction of peak to the GEMMs. Re-measure on H100 after tile re-tuning; the 32B config uses T = 4096 and
  head dim 128.
- **Fused CE [R25].** Cut Cross-Entropy, Liger kernels and our chunked CE all avoid full logits. At 32B the
  vocabulary share is small, so this matters mostly at 1B [E for CCE memory claim, per R25].
- **Packing with document masks.** Zhao et al. (2402.13991) [E]: causal masking across documents injects
  distracting context. Intra-document masking "significantly improves" results, and related-document packing
  helps more. **For us [S]:** without softmax there is no row normalization to dilute cross-document
  contributions. Leakage adds unnormalized signed terms growing with n_visible, so document masking is more
  important than for softmax. Our kernel's document-tile skipping already does this. Make sure the n^−½ scale
  counts **same-document** visible tokens.
- **FP8.** Float8 adds about 40–50% throughput in TorchTitan. Defer it for this architecture (§4.2).

---

## 7. Scaling-law methodology on small budgets

- **Porian et al. (NeurIPS 2024) [R19].** [E] The Kaplan vs Chinchilla gap closes after three fixes: counting
  head (unembedding) FLOPs (exponent 0.86 → 0.70), warmup tokens ≈ N (→ ~0.60), and per-size tuning of LR, batch
  and β2 (→ ~0.50). 16 models from 55M to 901M, more than 900 runs. **Relevance:** our 17% warmup at 4.8M-token
  rungs is the "warmup too long for small models" failure. Use warmup tokens ≈ N where that is ≥100 steps.
- **Choshen et al., "A Hitchhiker's Guide to Scaling Law Estimation" (2410.11840).** [E] About **5 models** is a
  safe minimum. Prefer more sizes over more seeds of one size for *fitting*. Fit on **intermediate checkpoints**,
  dropping early ones (first ~10B tokens at their scale) cuts error from >15% to 4–10%. The best achievable
  relative error is ~4%. Seed variance reaches 3.5%. Training the target for ~30% of its run gives good final-loss
  estimates.
- **DataDecide (Ai2, 2504.11393).** [E] Ranking at a single small size (150M) picks the 1B winner about **80%**
  of the time. None of 8 scaling-law fits beat that on the compute-decision frontier. Continuous likelihood metrics
  make downstream tasks predictable. 3 seeds, up to 100B tokens.
- **OlmPool (Ai2 2026) [R24].** [E] 26 × 7B models, 140B tokens each, extended to 64K. Training loss, perplexity
  and 16 short-context benchmarks failed to predict 32K–64K ability. Models near-identical on standard evals
  differed by more than 26 HELMET points. **QK-norm had the largest single effect**, and combined choices compounded
  to −47%. Rankings were stable from 140B tokens onward. An early extension probe is cheap. **Relevance:** if we
  adopt QK-norm for stability, pair it with a context-extension probe. Our no-softmax attention is itself a
  long-context risk (n_visible scaling) that short-context CE will not reveal.
- **Chinchilla replication (Besiroglu et al. 2404.10102)** found Chinchilla's parametric fit unreliable, with
  implausibly narrow CIs [U: not re-checked]. **Gemstones (2502.06857)** reports that scaling-law prescriptions are
  sensitive to the width/depth and LR choices of the fitted models [U].
- **Shen et al. 2406.16690.** [E] 1,376 intermediate checkpoints from 70M to 7B were used to fit laws for
  linear-complexity models. Intermediate checkpoints are standard practice.

**For us [S].** (1) The binding error bar is seed variance: 0.01–0.02 CE, and 0.06 in an unstable cell, against
paired effects of 0.02–0.15. Use 3 seeds for any claim below ~0.03 CE. (2) For the architecture-vs-softmax
scaling claim, train a matched ladder of at least 5 sizes (e.g. 50M, 100M, 200M, 500M, 1B) at fixed TPP, with
per-size LR from our transfer rule, plus a ±√2 check at two sizes. Fit on cooled endpoints, or on EWA'd
intermediate checkpoints validated against cooldowns. (3) Count FLOPs including unembedding and the polynomial
attention's real cost. Compare at equal **measured** GPU-hours as well as equal FLOPs. (4) Do not trust a
100M-token architecture ranking for 32B decisions. Promote with Hyperband-style survivors [R23], and keep a
"middling" candidate as DataDecide's ~20% failure insurance.

---

## Claims I could not verify from the papers (consolidated)
- Everett et al.: the exact empirical *global*-LR exponent for SP+Adam (tables garbled).
- Haas et al.: a specific n^−½ optimal exponent. The abstract says only "slower than theory".
- Complete^dP Table 1: β1/β2 and ε batch/duration scaling entries (garbled), and the 420× vs ~600× compute
  inconsistency.
- Qiu et al.: the "only √(d_out/d_in) transfers for Muon" detail comes from a search summary, not the paper body.
- Kimi K2: the QK-clip threshold (100) and the reason QK-norm was rejected (MLA) come from secondary sources.
- ScheduleFree+ claims (abstract and secondary summary only).
- Llama 3 MFU, DeepSeek and Step-Law exponents, Chinchilla-replication and Gemstones details, outlier-feature
  papers, Π-nets, and the specific output-norm placement in RetNet, GLA and MiniMax: from memory, not re-fetched.
- Everything marked [S], in particular the bilinear-specific theory in §1.7, the activation-memory and FLOP-share
  arithmetic in §6, and the claim that document masking matters more without softmax.

---

## References
- [R13] Yang et al., Tensor Programs V (μP), 2022. https://arxiv.org/abs/2203.03466
- [Everett24] Everett et al., Scaling Exponents Across Parameterizations and Optimizers, ICML 2024. https://arxiv.org/abs/2407.05872
- [Kosson25] Kosson et al., Weight Decay may matter more than μP for Learning Rate Transfer in Practice, ICLR 2026. https://arxiv.org/abs/2510.19093
- [Haas25] Haas et al., On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling, NeurIPS 2025. https://arxiv.org/abs/2505.22491
- [2510.15262] Robust Layerwise Scaling Rules by Proper Weight Decay Tuning. https://arxiv.org/abs/2510.15262
- [Kalra26] Kalra & Barkeshli, Quantifying Hyperparameter Transfer and the Importance of Embedding Layer Learning Rate, ICML 2026. https://arxiv.org/abs/2605.21486
- [uMuP] Blake et al., u-μP: The Unit-Scaled Maximal Update Parametrization, 2024. https://arxiv.org/abs/2407.17465
- [TP6] Yang et al., Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks, 2023. https://arxiv.org/abs/2310.02244
- Bordelon et al., Depthwise Hyperparameter Transfer in Residual Networks, 2023. https://arxiv.org/abs/2309.16620
- [CompleteP, R14] Dey et al., Don't be lazy: CompleteP, NeurIPS 2025. https://arxiv.org/abs/2505.01618
- [CdP, R15] Mlodozeniec et al., Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration. https://arxiv.org/abs/2512.22382
- [Zhang25CBS] Zhang et al., How Does Critical Batch Size Scale in Pre-training?, ICLR 2025. https://arxiv.org/abs/2410.21676
- [PowerLines] Bergsma et al., Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training, 2025. https://arxiv.org/abs/2505.13738
- [Merrill25] Merrill et al., Critical Batch Size Revisited, 2025. https://arxiv.org/abs/2505.23971
- [Marek25, R18] Marek et al., Small Batch Size Training for Language Models, NeurIPS 2025. https://arxiv.org/abs/2507.07101
- [Bjorck25, R16] Bjorck et al., Scaling Optimal LR Across Token Horizons, ICLR 2025. https://arxiv.org/abs/2409.19913
- Wang & Aitchison, How to set AdamW's weight decay as you scale model and dataset size, 2024. https://arxiv.org/abs/2405.13698
- DeepSeek LLM, 2024 [U]. https://arxiv.org/abs/2401.02954
- Li et al., Predictable Scale Part I (Step Law), 2025 [U]. https://arxiv.org/abs/2503.04715
- [Hagele24, R17] Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations, NeurIPS 2024. https://arxiv.org/abs/2405.18392
- [Bergsma25D2Z] Bergsma et al., Straight to Zero, ICLR 2025. https://arxiv.org/abs/2502.15938
- Song et al., Through the River (schedule-free), NeurIPS 2025. https://arxiv.org/abs/2507.09846
- ScheduleFree+ [U], 2026. https://arxiv.org/abs/2605.19095
- [Wortsman23] Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities, ICLR 2024. https://arxiv.org/abs/2309.14322
- [OLMo2] OLMo 2 Furious, 2025. https://arxiv.org/abs/2501.00656
- [R07] Zhai et al., σReparam, 2023. https://arxiv.org/abs/2303.06296
- [KimiK2] Kimi K2 technical report and blog. https://arxiv.org/abs/2507.20534 , https://www.kimi.com/blog/kimi-k2.html
- [Fishman24] Fishman et al., Scaling FP8 training to trillion-token LLMs, ICLR 2025. https://arxiv.org/abs/2409.12517
- He et al., Understanding and Minimising Outlier Features in Transformer Training [U]. https://arxiv.org/abs/2405.19279
- Sun et al., Massive Activations in LLMs [U]. https://arxiv.org/abs/2402.17762
- [R28] Manifest AI, Scaling Context Requires Rethinking Attention (power attention). https://arxiv.org/abs/2507.04239 ; kernels https://github.com/m-a-n-i-f-e-s-t/retention
- Kacham et al., PolySketchFormer, 2023. https://arxiv.org/abs/2310.01655
- Arora et al., Based, 2024. https://arxiv.org/abs/2402.18668
- Qin et al., The Devil in Linear Transformer, 2022. https://arxiv.org/abs/2210.10340
- MiniMax-01, 2025. https://arxiv.org/abs/2501.08313
- Sun et al., RetNet [U]. https://arxiv.org/abs/2307.08621 ; Yang et al., GLA [U]. https://arxiv.org/abs/2312.06635
- Shen et al., Scaling Laws for Linear Complexity Language Models, EMNLP 2024. https://arxiv.org/abs/2406.16690
- Shazeer, GLU Variants Improve Transformer, 2020. https://arxiv.org/abs/2002.05202
- [R03] Pearce et al., Bilinear MLPs enable weight-based mechanistic interpretability, 2024. https://arxiv.org/abs/2410.08417
- Zhuo et al., Polynomial Composition Activations (PolyCom), ICLR 2025. https://arxiv.org/abs/2411.03884
- Chrysos et al., Π-nets [U]. https://arxiv.org/abs/2003.03828
- [MuonScalable] Liu et al., Muon is Scalable for LLM Training, 2025. https://arxiv.org/abs/2502.16982
- [Wen25, R21] Wen et al., Fantastic Pretraining Optimizers and Where to Find Them, ICLR 2026. https://arxiv.org/abs/2509.02046
- [Qiu25] Qiu et al., Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales. https://arxiv.org/abs/2512.05620
- μP for more optimizers via spectral conditions, 2026 [U]. https://arxiv.org/abs/2602.20937
- [EssentialMuon] Essential AI, Practical Efficiency of Muon for Pretraining, 2025. https://arxiv.org/abs/2505.02222
- [R22] Khona et al., SOAP, Muon, and Beyond, 2026. https://arxiv.org/abs/2607.20548
- [TorchTitan] Liang et al., TorchTitan, ICLR 2025. https://arxiv.org/abs/2410.06511
- [R25] Wijmans et al., Cut Your Losses in Large-Vocabulary Language Models, ICLR 2025. https://arxiv.org/abs/2411.09009
- [R31] PyTorch FSDP / distributed checkpointing. https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
- [Zhao24] Zhao et al., Analysing the Impact of Sequence Composition on Language Model Pre-Training, 2024. https://arxiv.org/abs/2402.13991
- [R19, Porian24] Porian et al., Resolving Discrepancies in Compute-Optimal Scaling of Language Models, NeurIPS 2024. https://arxiv.org/abs/2406.19146
- [Hitchhiker] Choshen et al., A Hitchhiker's Guide to Scaling Law Estimation, 2024. https://arxiv.org/abs/2410.11840
- [DataDecide] Magnusson et al., DataDecide, 2025. https://arxiv.org/abs/2504.11393
- [OlmPool, R24] Ai2, OlmPool, 2026. https://allenai.org/blog/olmpool
- Besiroglu et al., Chinchilla Scaling: A Replication Attempt [U]. https://arxiv.org/abs/2404.10102
- McLeish et al., Gemstones [U]. https://arxiv.org/abs/2502.06857
- [R23] Li et al., Hyperband. https://www.jmlr.org/papers/v18/16-558.html
