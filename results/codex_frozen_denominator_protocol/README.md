# Frozen-denominator protocol evidence

CPU only, untrained two-layer eight-wide FP64 model; synthetic token IDs, not a corpus or GPT2 benchmark. No training, GPU, checkpoint or confirmation-directory access. This demonstrates an evaluation protocol and a counterexample to transferring small in-distribution approximation error; it establishes no language-model quality, real-domain OOD conclusion or learned known-mechanism capability.

Run from repository root:

```bash
PYTHONPATH=. /venv/main/bin/python runs/codex_frozen_denominator_protocol/reproduce.py
```

Four explicitly named calibration documents provide52 input positions. Two evaluation and two shifted-alphabet documents have disjoint IDs and full-content hashes from calibration and each other. This verifies exact whole-document inequality only; token substrings may overlap. Independent forward-prehooks capture each DynamicNorm denominator under the live model; arithmetic input-position means are fitted once per site. Constants and their tensor hashes are then reused unchanged for every condition. This never calls `analysis.fixed_denominator` or its batch-half split.

Each evaluation condition scores48 canonical source targets. Context8 and24 use exactly the same target offsets and distinct available-prefix lengths. Metrics are token-weighted. The controlled shift changes synthetic token support from1–7 to17–23; it is not a real-domain benchmark.

| Synthetic condition | Live CE | Fixed CE | KL(live‖fixed) | Mean absolute logit error |
| --- | ---: | ---: | ---: | ---: |
| Original support, context8 | 3.46730 | 3.47959 | .001421 | .03197 |
| Original support, context24 | 3.46368 | 3.47346 | .001889 | .03345 |
| Shifted support, context8 | 3.48619 | 1.253e15 | 9.031e14 | 4.145e14 |
| Shifted support, context24 | 3.48332 | 7.275e14 | 5.077e14 | 2.329e14 |

The shifted fixed model remains finite in FP64 but amplifies polynomial activations severely. These deliberately synthetic results illustrate why small original-distribution KL cannot establish shifted-input equivalence. Two exact-label Boolean prompts additionally exercise answer-only live/fixed CE and two-candidate forced-choice correctness (not full-vocabulary/free-generation accuracy); the model is random/untrained, so those observations are numerical sanity probes, not task proficiency or composition generalization.

Per-token calibration replay with captured denominators clamped reproduces live logits exactly (max error0). Assertions verify immutable constants through all probes, unchanged model parameters/buffers and Torch RNG, and exact restoration of heterogeneous per-module training modes plus every norm mode/saved-denominator/constant attribute. `evidence.json` includes constants, document tokens/IDs/content hashes, target-plan hashes, exact metrics and source/script hashes.

Requested official API: separate explicitly identified calibration/evaluation documents; a serialized per-site frozen-constant artifact with fit provenance/hash; reuse the same constants across context/domain/task conditions; report source-offset overlap, target counts, finite/overflow outcomes and restoration checks. Keep per-token clamped replay distinct from the fixed-statistic approximation, and retain explicit empirical scope rather than infer polynomial/OOD equivalence from small average error.
