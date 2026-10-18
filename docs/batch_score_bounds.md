# Training-batch score bounds and QK clipping

`train.qk_clip_source` selects `probe` (default, existing behavior) or
`batch_bound`. With `batch_bound`, polynomial attention records per-head
Cauchy–Schwarz score bounds during every training forward, including the Triton
backend. No extra probe forward is needed for this mode.

Every update logs `score_bound_max`: the maximum over all executed microbatches,
layers, factors and heads. Distributed recording reduces that maximum over ranks,
including ranks with zero selected prediction targets. Bounds cover executed
forward inputs, including padding and unscored positions; they are intentionally
looser than a masked score maximum. Softmax attention does not provide these
bounds and rejects this source selection.

`qk_clip_tau: 0` permits recording without clipping. With positive tau, the
existing `qk_clip_every_steps` global cadence applies. After the optimizer update
and normalization commits, the serial trainer calls `qk_clip_from_bounds_` using
only the final microbatch's **original forward** bounds, as requested. It retains
that snapshot across activation recomputation and applies clipping before the
optional stability probe and checkpoint saves. Optimizer moments and counters
are unchanged. Probe diagnostics preserve recording flags and cached bounds.

These are pre-update activation measurements. They do not certify a post-update
score cap: the optimizer has changed weights, and clipping an upstream layer can
change downstream inputs. The per-update maximum can therefore exceed the
`qk_clip.max_bound_before` value, which describes only the final microbatch.
Malformed or nonfinite recorded bounds abort before an optimizer update. Negative
bounds also fail validation. Distributed failures are coordinated across ranks.

Recording-only mode works in serial and distributed training. Active clipping
still supports serial plain-linear Q/K models only; spectral parameterizations
and distributed clipping retain their existing rejection. Source selection is
part of checkpoint identity. Legacy checkpoints missing the field can resume
with `probe`, but cannot silently switch to `batch_bound`.

CPU tests cover actual compiled forwards (`aot_eager`), activation recomputation,
exact serial resume and cadence, independent-probe avoidance, maxima from earlier
microbatches, probe state preservation, and two-rank global maxima/nonfinite
failure handling. CUDA/Triton execution and multi-H100 throughput are not measured
by these trainer tests.
