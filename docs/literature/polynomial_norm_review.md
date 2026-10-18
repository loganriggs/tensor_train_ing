# Literature review: removing input-dependent normalization (polynomial-at-inference)

**Sources and access.** Both papers were read through WebFetch, which returns model-extracted text and LaTeX, not raw PDF. TaperNorm: arXiv 2602.10408v2 HTML. Polynomial NN framework: arXiv 2505.11589v1 abs and HTML. Equations below are as extracted from those pages. Caveats:
- TaperNorm Table 1 row labels came through garbled (see below).
- No ablation numbers for the no-aux-loss runs could be found.
- Model sizes for 2505.11589 are not reported in the paper.

---

## 1. TaperNorm: "Gated Normalization Removal and Scale Anchoring in Pre-Norm Transformers" (2602.10408)

### Definition
RMSNorm variant:

$$\mathrm{TaperNorm}(h;g)=g\,\frac{h}{r(h)}D_\gamma+(1-g)\,c\,h\,D_{\tilde\gamma},\qquad r(h)=\sqrt{\|h\|_2^2/d+\varepsilon}$$

- $c$ is a per-layer scalar.
- $\tilde\gamma$ is a per-feature gain.
- $g=1$ is RMSNorm. $g=0$ is "a sample-independent scaling that can be fused into downstream linear operations."

LayerNorm variant (App. B):

$$\beta+g\frac{h-\mu_h}{\sigma_h}D_\gamma+(1-g)c(h-\mu_h)D_{\tilde\gamma}$$

**Calibration of c (Sec. 3.4).** $c$ is a least-squares match of the linear branch to the normalized branch:

$$c^\star=\frac{\mathbb{E}[\|hD_\gamma\|_2^2/r(h)]}{\mathbb{E}[\|hD_\gamma\|_2^2]}$$

- Numerator and denominator are EMAs, updated as $s\leftarrow(1-\mu)s+\mu x_k$ with $s_0=0$.
- Bias correction divides by $1-(1-\mu)^n$, then $c^\star=\hat s_{num}/(\hat s_{den}+\delta)$.

**Schedule.**
1. During LR warmup, keep $g=1$ and accumulate the EMAs.
2. At taper start, freeze $c$ and set $\tilde\gamma\leftarrow\gamma$. $\gamma$ and $\tilde\gamma$ both stay trainable.
3. Cosine-decay one shared gate $g$ from 1 to 0, applied to all tapered layers.
   - TinyStories: taper start = end of the 5% warmup; the decay runs "over the remaining training steps."
   - GPT-2 fine-tuning (total / warmup end / taper start / taper end): S 300/25/25/100, M 500/10/20/200, L 600/30/30/500, XL 800/20/50/700.

**Inference map.** At $g=0$ the layer is $c\,h\,D_{\tilde\gamma}$, folded into the next linear layer as $W\leftarrow cD_{\tilde\gamma}W$.

**Variants.**
- **Internal-Taper** tapers only the block norms. The final RMSNorm stays an ordinary RMSNorm.
- **All-Taper** also tapers the final norm, giving a fully norm-free model. The paper calls it "more fragile."

**Scale anchoring (Sec. 3.5).**

$$\mathcal{L}_{aux}=\lambda\,\mathbb{E}_{b,t}(s(h_{b,t})-s_{tgt})^2,\qquad \mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{aux}$$

- $s=r(h)$ for RMSNorm, or $\sigma_h$ for LayerNorm.
- $s_{tgt}$ is the bias-corrected warmup EMA, frozen at taper start.
- I did not confirm exactly which $h$ (which layers) the loss is applied to; it appears to be the input to the final norm.

**Theory.**
- Prop 4.1: a 0-homogeneous final norm gives zero radial gradient.
- Prop 4.2: without it, $\langle\nabla_h\ell,h\rangle\le-(1-\mathrm{softmax}(z)_y)m<0$, a radial push the paper calls "logit chasing."
- Prop 4.3: the aux loss adds a radial restoring force, $\langle\nabla_h\mathcal L_{aux},h\rangle=\frac{2\lambda(r(h)-s_{tgt})}{d\,r(h)}\|h\|_2^2$.

**Hyperparameters.**
- TinyStories: AdamW (0.9, 0.95), LR 3e-4, 5% warmup, cosine to 0, wd 0, batch 16 × 512, clip 1.0, $\mu=0.01$, $\lambda=0.1$, 6 seeds. Training ran 500k steps (d = 64, 128), 750k (d = 256) and 1M (d = 512).
- GPT-2: $\mu=0.1$; $\lambda=0.1$ for S/M/L and 0.01 for XL.

### Exactly sample-independent at inference?
- **All-Taper: yes.** Every norm becomes a fixed diagonal scale that folds into the weights.
- **Internal-Taper: no.** It keeps the final RMSNorm, which is a single input-dependent denominator before the unembedding.
- Their attention is softmax, so their models are not polynomial anyway. Only the normalization part transfers to us.

### Evidence
- **Pretraining.** TinyStories, depth 8, SwiGLU, RoPE, tied embeddings. Validation loss, baseline → Internal-Taper → All-Taper:
  - 2.1538 → 2.1852 → 2.1930
  - 1.7561 → 1.7828 → 1.7874
  - 1.4620 → 1.4837 → 1.4837
  - 1.2768 → 1.2859 → 1.2868

  The penalty is about +0.7–1.8%. Rows are labelled 11M / 33M / 99M / "3030M" on the page (garbled). The text says pretraining went "up to about 30M," so the true sizes are unclear.
- **Fine-tuning.** GPT-2 S–XL norm removal, OpenWebText CE:
  - S: 3.0126 → 3.0584 (Internal) / 3.0685 (All)
  - XL: 2.4799 → 2.5144 / 2.5305

  This is comparable to FakeLN.
- **Decoding speed.** 1.15–1.18× with folding on H100.
- **Failures.**
  - Without the aux loss, removing the final norm makes logit norms grow and gradients diverge.
  - Removing all LNs at once destabilizes training (prior work).
  - Untested beyond about 30M parameters / depth 8 for pretraining.
  - Robustness to activation outliers is open.

### Mapping onto `x + alpha*F(norm_in(x))`
Changes to the training loop:
1. Make `norm_in` = $g\cdot\mathrm{RMSNorm}_\gamma(x)+(1-g)\,c\,x\odot\tilde\gamma$.
2. Add per-layer EMA buffers for $\mathbb E\|xD_\gamma\|^2/r$, $\mathbb E\|xD_\gamma\|^2$ and $\mathbb E\,r(x)$, plus a step counter for bias correction.
3. At taper start: freeze $c$, copy $\gamma\to\tilde\gamma$ (a new parameter), and freeze $s_{tgt}$.
4. Run a global cosine $g(k)$ schedule. Add $\lambda(r(h)-s_{tgt})^2$ to the loss.
5. At the end, fold $c\tilde\gamma$ into L/R/Q/K/V.

Relation to our "lagged" package [inference]:
- Lagged is essentially $1/\mathrm{EMA}(r)$, a per-layer scalar like $c$ but kept moving.
- TaperNorm's extra pieces are: (a) a smooth $g$ interpolation from true RMSNorm instead of switching at once; (b) a frozen $c$ and $s_{tgt}$ anchored by an explicit loss on $r(h)$.
- Lagged + aux loss would be a cheap ablation.

Interaction with degree-5 attention [inference]:
- RMSNorm makes each sublayer 0-homogeneous in $x$. With a linear norm, the bilinear MLP scales as $|x|^2$ and the attention as $|x|^5$ (QK·QK·V, all from the normalized input).
- So any drift in residual scale is amplified 5× in log-scale. That makes the scale anchor, together with sigmaReparam bounding W, much more important for us than for softmax models.
- $s_{tgt}$ should probably anchor every block input, not only the final one.
- The final norm must also be tapered (All-Taper) for exact polynomiality. That is the fragile case, so expect to need $\lambda$ tuning, and probably a longer taper window than 5%→end.

---

## 2. "A Training Framework for Optimal and Stable Training of Polynomial Neural Networks" (2505.11589, Al Hossain & Rahman)

### Definition
- **Activation.** $f(x)=\sum_{i=0}^n a_ix^i$, initialized by a least-squares fit to ReLU on $m$ uniform points in $[-B,B]$. Coefficient LR $=0.1\times$ base LR.
- **Boundary loss.**
  $$L_i=\frac{1}{|X_i|}\sum_j\left(e^{\max(|X_{i,j}|-\alpha B,0)}-1\right),\qquad L_{total}=L_{CE}+\lambda\sum_iL_i$$
- **Selective gradient clipping.**
  $$\nabla\theta'_{non\text{-}BN}=\nabla\theta_{non\text{-}BN}/\max(1,\|\nabla\theta_{non\text{-}BN}\|_2/c)$$
  BatchNorm $\gamma,\beta$ gradients are not clipped. Clipping them "causes training collapse at degree 8."

**Hyperparameters.**
- $\lambda=1000$; robust over 1–1000.
- $\alpha$ = 1 / 0.75 / 0.5 for degree 2 / 4 / ≥8.
- $B$ = 10–14 (degree 2), 20–30 (degree 4), 35 (degree 8).
- Clip $c$ swept 0.5–1.0.
- AdamW, LR 1e-3, Kaiming-uniform init.

### Exactly polynomial at inference?
**Yes**, but trivially so. The normalization is **BatchNorm**, which at inference uses running statistics and is "fused into adjacent linear layers as affine transformations." Any CNN with BatchNorm and polynomial activations is polynomial at inference. The paper does not deal with LayerNorm/RMSNorm or with transformers.

### Evidence
- ResNet-18 (average pooling) on MNIST, FashionMNIST, CIFAR-10/100 and Speech Commands; a 7-layer MLP on UCI-HAR and Capture-24. Model sizes are not reported.
- Degree 2 vs ReLU:
  - CIFAR-10: 0.89 vs 0.91
  - CIFAR-100: 0.63 vs 0.64
- Degree 4–8 matches ReLU (e.g. CIFAR-100: 0.66 at degree 8).
- Baseline, boundary-loss-only and clip-only runs are often unstable at higher degree (p < 0.01).
- Very small $B$ destabilizes training; very large $B$ hurts accuracy.
- Degree 22 is claimed stable, but no accuracy is given.
- Encrypted (HE) inference on UCI-HAR: 94.7%.

### Mapping to us [inference]
- **BatchNorm.** BN statistics computed over the batch are the other route to a sample-independent map at inference. Our "lagged" EMA is the token-wise analogue.
- **Boundary loss.** Directly reusable. Put an exponential hinge penalty on the inputs to the bilinear MLP and on the QK scores, with a bound $B$, to keep degree-5 terms from blowing up. This complements the TaperNorm scale anchor (the boundary loss is elementwise; TaperNorm's anchor is on the RMS).
- **Selective clipping.** Analogue for us: exempt gain/norm parameters ($\gamma,\tilde\gamma$, sigmaReparam scales) from global clipping.

---

## 3. Related work yielding sample-independent / norm-free networks (from the papers)

**From TaperNorm:**
- Baroni et al. 2025, FakeLN, which removes GPT-2's LayerNorms sequentially at inference: **2507.02559** (the closest prior work)
- σReparam: 2303.06296
- DyT (elementwise tanh; not polynomial): 2503.10622
- Derf: 2512.10938
- Fixup: 1901.09321
- ReZero: 2003.04887
- LayerScale: 2103.17239
- DeepNet: 2203.00555
- NF-Nets: 2102.06171
- Deep Kernel Shaping: 2110.01765
- Transformers without shortcuts: 2302.10322
- LogitNorm (output-scale control): 2205.09310

**From 2505.11589:**
- AESPA: 2201.06699
- Layerwise polynomial approximation: 2310.10349
- Approximate activations for HE: 2011.05530
- Input stabilization for HE: 1902.01870
- CryptoDL: 1711.05189
- HE tuning of large CNNs: 2304.14836
- Polynomial interpretability (Dubey et al.): 2205.14108
- Zimerman et al. 2023, "converting transformers to polynomial form": cited without an arXiv ID; not verified.
