# Periodic training stability diagnostics

Serial training supports the opt-in `train.stability_every_steps` interval (strict nonnegative integer; zero is off). Claude added `stability_every_steps: int = 0` to TrainConfig. YAML activation is now verified with real CPU pre-RMS and lagged training plus exact split/resume checks.

Every N completed global optimizer updates, after lagged-normalization commit, the trainer calls `tnlm.analysis.stability_stats` on the raw eager model. A separate train-only sampler supplies a fixed first row using the configured training seed and a context capped at512 tokens. It never advances the training sampler or reads validation/test data. Resume recreates the same probe for unchanged data/seed/context. A context fork can change the probe, with that change visible in its identity.

Sampled `metrics.jsonl` rows contain:

- `stability.stats`: per-layer attention-score and bilinear-product statistics, plus the diagnostic's worst-case summary.
- `stability.probe`: batch hash, manifest hash, split, seed, context and batch size.
- `stability.seconds`: diagnostic elapsed time.
- `stability.nonfinite_paths`: locations whose NaN/Inf values were converted to JSON null. Nonfinite diagnostic results remain visible rather than invalidating the metrics JSON.

The helper uses no gradients, eval mode, and disabled lagged-statistic collection. It restores all module training flags, lagged collection/validity state, and Python/NumPy/Torch RNG state, including on exceptions. No model/optimizer-sized copies are made. Normal diagnostic failures propagate after restoration; logging is not silently skipped. The Claude-owned diagnostic retains ownership of its mathematical formulas and read-only implementation.

The eager diagnostic runs outside BF16 autocast on the model's parameter dtype. It is a consistent monitoring probe, not an exact reconstruction of training's BF16 intermediate values. The model forward and materialized attention scores incur extra compute and memory; overhead at1B has not been GPU-measured. Diagnostic time is included in training step time, billing and throughput-canary measurements. An enabled flag can therefore lower measured throughput. Zero avoids constructing the probe entirely.

Distributed opt-in explicitly rejects before training because this diagnostic has not been made safe for sharded collective forwards. Single-device compiled training uses the underlying eager model for the probe, avoiding diagnostic hooks on the compiled training callable. No spectral update or optimizer step is performed by the probe.

CPU helper tests cover real pre-RMS and lagged models, fixed-probe identity, mixed module modes, RNG/lifecycle restoration, exception cleanup and nonfinite serialization. Actual serial training with periodic logging preserves model, optimizer, sampler and progress exactly relative to logging-off training, and produces the same probe statistics after a split/resume. No new GPU job or active sweep modification was made.

Integrated gate: **859 CPU tests passed**, three warnings, xdist8 with CUDA hidden, excluding kernel/parallel files. This also covers the screening retention changes. That initial gate preceded the schema addition.

Schema follow-up:26focused stability and legacy-compatibility tests pass. The real training/replay cases now load the enabled interval through YAML; probe identity/statistics and unchanged training trajectory are checked. Claude reports the12-candidate1B sweep is chained after joint50m. No chain, snapshot, config, or GPU process was changed by this verification.
