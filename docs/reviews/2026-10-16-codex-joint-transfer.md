# Independent joint width/depth pilot audit

The completed 24×1024 pilot supports a combined-package benefit at the middle and high tested learning rates. It does not establish an interaction law, a continuous learning-rate optimum, or interpretation preservation.

The audit copies and hashes nine evaluation/result pairs and available run configurations into `runs/codex_joint_transfer_audit/evidence/inputs/`. It reads no checkpoint tensor state or corpus tokens. The six joint runs have checkpoint-derived descriptors; some predate full configuration descriptors. The three 12×512 baseline reports lack descriptors. Their training progress and companion YAML are retained as weaker evidence, never upgraded to checkpoint-verified configuration identity. Every row records its evidence source.

All reports record 4,792,320 training targets, 585 updates and 100 warmup updates (17.09%); they are short uncooled pilots, slightly below the declared 4.8M target. Run metadata gives seed 101. Each evaluation uses the same 38,131 validation targets from 81 documents and matching available prefixes. Existing compare_evaluations validates target plans, frozen state, document separation, per-document aggregates and prefix identity before computing contrasts. Missing legacy training dimensions remain unknown.

| Base LR | Combined scaling minus default CE | 95% paired document interval |
| --- | ---: | ---: |
| 0.00015 | +0.095078 | [+0.086144, +0.104860] |
| 0.00045 | −0.096578 | [−0.109419, −0.084504] |
| 0.00135 | −0.217077 | [−0.239357, −0.199232] |

Intervals use 10,000 paired document resamples, seed 101. They characterize validation-document variation conditional on these trained models, not training-seed variability, repeated hyperparameter selection, or unseen-domain behavior.

At LR 0.00045, default 24×1024 improves CE by 0.042615 relative to the 12×512 baseline; combined scaling improves it by 0.139193. The ratio 0.3062 is arithmetic for this comparison, not an estimate of recovered model capacity. Both combined scaling and the default have their best observed result at 0.00045 on the three-point grid. The grid cannot show that this LR lies beyond the continuous optimum or identify the cause of the gap.

Increasing base LR from 0.00045 to 0.00135 raises CE by 0.298323 under the default and by 0.177824 under combined scaling. The paired four-report contrast

`(scaled_high − scaled_mid) − (default_high − default_mid)`

is −0.120499, with a 95% document-bootstrap interval [−0.142588, −0.101611]. Each draw resamples the same original document indices across all four reports, then divides the weighted loss-sum contrast by the resampled target count. Subtracting two marginal interval endpoints would not account for their covariance. This is an exploratory LR-sensitivity contrast, not a width-by-depth factorial interaction.

Width-only and depth-only scaling cells at **24×1024** are absent. Earlier width-only 12×1024 and depth-only 24×512 runs do not fill those cells: architecture changes with the intervention. Width scaling changes hidden learning rates and logit scale; the depth option changes both residual multipliers and output initialization. The pilot establishes a benefit of the combined package at two tested LRs, not isolated causality of any one component. It worsens the low-LR result.

Suggested correction to Claude's scaling narrative: “At the doubled width and depth, the combined scaling package improves validation CE at the transferred middle LR and reduces the penalty at the higher tested LR in this short one-seed pilot. Coarse LR spacing, missing same-shape factorial controls, and absent cooled multi-seed evidence limit conclusions.” Claims that the effect compounds at every doubling or that the default is already past its optimum need additional evidence.

Artifacts: `summary.json` contains 12 matched comparisons and source hashes; `high_lr_contrast.json` contains the four-report contrast; `joint-transfer.svg` plots CE and paired differences. Reproduction scripts are adjacent to `evidence/` and intentionally refuse to overwrite that directory. Earlier incomplete audit attempts are labeled separately and are not evidence. No production edits, GPU jobs, final-test probes or modifications to Claude's experiments.
