# Public analysis lifecycle audit

CPU fault injection against `tnlm.analysis.analyze` confirms that failure does not preserve the caller's module state. Source SHA256 is recorded in `runs/codex_analysis_lifecycle_audit/evidence.json`; the script reproduces all cases without modifying production code.

The caller starts with a training model and an evaluation-mode final norm. Three cases are tested: empty probe input, a failure in model forward, and a failure in the local-tensor diagnostic after capture. All three leave previously training modules in evaluation mode. Forward/diagnostic failures additionally clear previously stored denominator tensors at all three dynamic norm sites. Parameters and registered buffers remain identical; forward/pre-forward hooks are restored.

The cause is visible in `analyze`: it calls `model.eval()` before validating probes, and the final `model.train(was_training)` executes only on success. Even on success that recursive call cannot reconstruct heterogeneous per-module modes. `_modes` resets denominator attributes and ablation flags to defaults rather than restoring caller values.

Requested correction in Claude-owned analysis: snapshot each module's training flag and relevant intervention attributes before mutation, then restore them in an exception-safe outer context covering probe conversion/validation and all diagnostics. If existing active interventions are unsupported, reject them before mutation; do not silently reinterpret an already-intervened model. Preserve original tensor references where the code only replaces attributes. Parameter/buffer mutation is outside this proposed lightweight lifecycle guard and should continue to fail validation.

This is a public function reuse issue. The CLI constructs a fresh model for each invocation, checks successful diagnostic lifecycle integrity, and bills failures. The evidence does not show reuse of a contaminated model across CLI invocations or corruption of saved checkpoints. No GPU, training, corpus or confirmation-directory access was used.

Independent read-only review confirmed the current source hash, all three failure-path conclusions and the CLI scope distinction. Requested regression cases include successful heterogeneous-mode restoration as well as empty/forward/diagnostic failures. Inactive cached denominator attributes need not be rejected; preserve them.
