# Paired training-seed evidence

Brief §6 requires replication when candidate improvements are small. Existing
paired-document intervals describe variation across held-out documents for fixed
trained models. They do not estimate training-seed variation. The new
`compare_seed_pairs` command reads explicitly paired evaluation reports and keeps
those sampling units separate.

For each declared seed, define `d_s = CE_right,s - CE_left,s` on the same frozen
validation targets and available prefixes. Each training replicate receives
equal weight in the mean difference. The sample standard deviation uses the
centered differences with divisor `S - 1`; the standard error is that deviation
divided by `sqrt(S)`. These are the usual paired-observation summary formulas in
the [NIST statistical handbook](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).

With two pairs, `SE = abs(d_1 - d_2) / 2`. This can be very unstable, including
zero when the two observed differences happen to agree. No seed confidence
interval, p-value, automatic winner or HPO correction is produced. The SE has its
usual interpretation only for independent training replicates sampled from the
intended procedure; report paths and seed labels cannot establish that assumption.
Equal numeric seeds across architectures also do not prove coupled random streams.

Do not aggregate the sweep's existing single-reference comparisons into paired
seed evidence: each seed must pair the two conditions explicitly. The command
checks complete training segments, unique declared seeds, distinct checkpoint
identities within each condition, matching validation targets/prefixes, and
condition consistency. It does not prove that the pair list or allowed differences
were registered before observing the results. Preserve prospective experiment
plans separately. Exact-document separation checks do not prove near-duplicate
independence.

Checkpoint-derived descriptors now retain the full configuration except project
execution/billing settings. Within a condition it must remain identical except
for the training seed; this catches architecture or optimization differences
omitted by shorter summaries. Between conditions every configuration difference
must be declared, including differences not represented by summary dimensions.
The raw input report hashes and per-pair contrasts remain in the resulting report.

Example pair file (paths are relative to this YAML file):

```yaml
- seed: 101
  left: baseline_101/evaluation/evaluation.json
  right: candidate_101/evaluation/evaluation.json
- seed: 202
  left: baseline_202/evaluation/evaluation.json
  right: candidate_202/evaluation/evaluation.json
```

For a declared LR contrast:

```bash
tnlm compare_seed_pairs --pairs seed_pairs.yaml --out NEW_COMPARISON --allow-differences lr
```

The report is written to `NEW_COMPARISON/replication.json`. Existing output
directories are rejected. Summary names such as `lr`, `width` and `normalization`
also authorize their directly corresponding configuration fields. Other changes
need explicit paths, for example `--allow-differences width config.model.heads
config.model.d_mlp`. A path does not authorize unrelated summary differences.
All reports must use the same training manifest, validation targets, evaluation
seed, context and available prefixes. Missing legacy configuration or context
history prevents this strict summary; regenerate reports only from checkpoints
that actually retain the required evidence, never infer missing history.

Duplicate checkpoint state/manifest identities within a condition are rejected.
Equal model-weight outcomes with distinct checkpoint identities remain valid
observations and are reported diagnostically; requiring distinct outcomes would
select on the quantity being measured. No checkpoint files are reopened: hashes
bind the supplied reports and their stated checkpoint identities, not independent
verification of checkpoint contents or runtime/code equality.

Validation: 17 focused tests cover independent paired-statistic expectations,
immutable input hashes, rejected identity/configuration/target mismatches, equal
model outcomes and four real CPU train/evaluation runs (two seeds × two LR
conditions). CLI path resolution also passes. The full CPU suite passed **574
tests in 157.15 seconds**, with three warnings and no skips; CUDA kernel/parallel
test files were excluded. This validates the reporting path, not an empirical
winner among the research architectures.
