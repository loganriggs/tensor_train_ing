# Fixed-cadence QK clipping in training

The serial trainer calls Claude's `tnlm.qk_clip.qk_clip_` after optimizer and
normalization updates, before stability diagnostics and checkpoint saves.

Confirmed TrainConfig names (schema owned by Claude):

```yaml
train:
  qk_clip_tau: 100.0
  qk_clip_every_steps: 50
```

Defaults are `0.0` and `0`. Tau must be finite and nonnegative; zero disables
clipping. Cadence must be a nonnegative integer and must be positive when tau is
positive. Enabled distributed and spectral configurations reject before training.
The trainer supports these attributes now; YAML loading requires Claude's two
TrainConfig fields to land.

Cadence uses cumulative optimizer steps, so resumed runs retain the same clipping
boundaries. The fixed probe is shared with stability logging: an independent
train-only sampler supplies one row, capped at 512 tokens. Probe construction and
execution do not advance the training sampler. Execution preserves Python, NumPy
and Torch RNG, module modes, and lagged-normalization collection state. Clipping
also works with stability logging disabled; when both are disabled no probe is
allocated.

Clipping intentionally changes Q/K parameters. Optimizer moments and step counters
remain unchanged. Saved config pins both controls on ordinary resume, and saved
model state includes clipping. Older missing fields are compatible only with the
disabled defaults. A new policy cannot silently be enabled on ordinary resume.

On clipping steps, `metrics.jsonl` contains `qk_clip` with the helper's return
values, probe identity, elapsed seconds and optimizer-state policy. Probe time is
included in step timing. Helper errors or nonfinite returned statistics fail the
run; lifecycle state is restored, but partially mutated weights are not rolled
back or saved as a new successful checkpoint.

The score bound is measured on this fixed probe, not all training or unseen data.
The current helper performs a full eager probe forward per layer. GPU overhead
has not been measured and no active GPU queue or source snapshot was modified.

CPU tests cover real pre-RMS and lagged clipping, cumulative cadence, exact
split/resume model/optimizer/sampler/progress/RNG, unchanged optimizer moments,
clipping before diagnostics, default-off allocation, invalid controls and resume
policy changes, and lifecycle restoration on exceptions/nonfinite results.

Validation: 46 focused CPU tests passed. The broad CPU suite (CUDA hidden,
xdist8, kernel/parallel files excluded) had 924 passes and three sweep source-freeze
failures caused by a concurrent docstring edit. After edits stopped, all 19 sweep
canary/retention tests passed, including those three failures.
