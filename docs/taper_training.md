# TaperNorm training lifecycle

Set `model.normalization: taper` and `model.final_normalization: taper` for an
all-Taper model. The trainer supports serial and FSDP training, with the same
checkpoint format. `train.compile` must be false and activation checkpointing
must be disabled because warmup and anchor collection mutate module state.

The trainer collects nonpadding activation statistics during the original LR
warmup and commits them once per optimizer update, globally reduced under FSDP.
It freezes the scale and anchor target at the end of warmup. The cosine gate uses
cumulative optimizer steps, including separate cooldown updates, and ends at
`round(train.taper_end_fraction * total_steps)`. The default fraction is 1.0;
the endpoint must be strictly after warmup. Full-rung steps are computed from the
configured token horizon and global prediction-token batch size.

After freezing, each microbatch's anchor is measured on selected prediction
positions and weighted by their count, just like summed cross-entropy. The
backpropagated objective includes `train.taper_anchor_lambda * anchor` (default
0.1). The `loss` metric remains cross-entropy. Metrics also contain `taper_anchor`,
`taper_gate_used` for the update, `taper_gate` for the saved post-update state,
and named `taper_constants` (`c`, `s_tgt`) once frozen per trainer invocation.
Stability and QK-clip probes do not contribute warmup statistics.

A completed full rung saves gate exactly zero. An all-Taper model with otherwise
polynomial operators then reports operator class P. A partial execution stopped
by a runtime or step cap can still have a nonzero gate, even in its `final/`
checkpoint. Restores retain the original warmup, token horizon and global step;
ordinary horizon promotion and batch-size branches are rejected for Taper because
they would change the gate schedule. Evaluation uses the checkpoint's gate.

CPU validation covers real serial warmup, token-weighted accumulation, partial
updates, exact resume, cooldown, and two-rank FSDP with uneven and zero local
quotas. GPU throughput and multi-H100 execution remain unmeasured.
