# Batched saved-evaluation comparisons

`compare_evaluations --pairs` runs explicitly named comparisons from saved validation reports. It performs no model inference or training. The existing single-pair command remains available.

Example `pairs.yaml`:

```yaml
- name: wider versus baseline
  left: evaluations/baseline/evaluation.json
  right: evaluations/wider/evaluation.json
- name: scaled versus conventional
  left: evaluations/conventional/evaluation.json
  right: evaluations/scaled/evaluation.json
```

```bash
python -m tnlm.cli compare_evaluations --pairs pairs.yaml \
  --out comparisons --bootstrap-samples 10000 --seed 101
```

Relative report paths resolve from the pair file's directory. Each row requires exactly `name`, `left` and `right`; names must be unique nonempty labels. Names never become filesystem paths. `--pairs` cannot be combined with `--left` or `--right`.

The backend reads each resolved report path once and records its exact byte SHA256 and length. It preserves declared pair order and delegates validation/statistics to `compare_evaluations`. All input comparisons and JSON serialization must succeed before the output directory is created exclusively. Existing output is never overwritten. The single output `comparisons.json` contains each comparison and the input hashes; it is published through a temporary file and rename.

For each pair, the point estimate is the sum of right-minus-left document loss sums divided by the common total target count. Each bootstrap draw resamples the same document indices on both sides, then recomputes that ratio with the sampled token counts. Documents with several evaluation windows have their sums/counts combined before comparison. This differs from an unweighted average of document CE differences when document lengths differ.

The command does not pool pairs, select winners or correct for multiple comparisons. Its intervals describe nominal document uncertainty, not training-seed variability or hyperparameter-search uncertainty. Reusing a bootstrap seed across pairs does not produce a joint inference or simultaneous interval. Use the separate `compare_seed_pairs` command for explicit replicated-seed summaries. Input report hashes identify the bytes used; they do not independently authenticate checkpoints or certify experiment independence.

The unrelated `analysis.py` ablation helper still aggregates fragments separately in the inspected source; its correction remains assigned to Claude. This command reuses the evaluation path, which aggregates by original document ID.

Validation: **69 tests passed in 11.38 seconds**, covering batch API/CLI, the existing evaluator and paired-seed summaries. An independent deterministic four-draw test checks unequal document sizes 1 and9, per-document summed-loss differences +1 and−9: point delta−0.8 and bootstrap ratios1/−0.8/−0.8/−1, with the configured linear percentile endpoints−0.985 and0.865.

The real-report CLI smoke test is saved under `runs/codex_batch_comparison_audit/final-output/`. Its three full comparison objects match the prior joint-transfer audit exactly, including 10,000-draw intervals at seed101; all six input byte hashes were rechecked. `verification.json` records source hashes. The earlier `output/` smoke artifact predates a scope-wording correction and is not the final evidence. No inference, GPU use or confirmation-directory access occurred.
