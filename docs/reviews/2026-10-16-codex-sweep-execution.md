# Sequential confirmation sweeps

The `sweep` command accepts the same explicit candidate list as `plan_search --candidates`. Each candidate carries its own resolved model, optimizer, data and project configuration.

```sh
python -m tnlm.cli sweep --config BASE.yaml --candidates GRID.yaml \
  --out runs/confirmation --max-seconds 1200 \
  --evaluation-max-seconds 600 --device cuda
```

Configuration dry-run remains the default. Planning does not start training. With execution enabled, the default sequence is a complete uncooled parent, a separate cooldown from the saved90% branch, then held-out validation. `--no-cooldown` explicitly requests evaluation of uncooled endpoints. The two protocols must not be ranked together as if schedule completion were matched.

The full uncooled parent stays available for continuation. The cooldown uses the actual branch token count and exact configured endpoint, including a partial last update. Training and evaluation use their existing spend-ledger accounting; the sweep does not add a second runtime charge. Initialization, checkpointing and evaluation still consume time, so nominal training-token throughput alone is not the complete dollar budget. The plan also reports rate multiplied by all declared stage time caps, and the corresponding margin-adjusted reservations. These are cap-based estimates, not guaranteed upper bounds: a step or checkpoint can overrun a deadline. Orchestration, integrity hashing and summary overhead are currently outside the stage timers and must be reconciled against provider billing.

A sweep journal freezes candidates, data-manifest hashes, execution controls and Python source hashes, including kernels. Source identities are checked before new charged stages and after stage completion; long sweeps need a stable source checkout. Completed stages are reused only after artifact verification. An interrupted or failed charged stage requires explicit recovery; rerunning the command does not silently allocate a new identity and restart it. Checkpoints are retained, and aggregate disk admission precedes execution. Admission is advisory rather than a reservation against concurrent writers.

This executor coordinates serial model training. It is not a multi-node launcher or an automatic successive-halving search policy. Candidate quality comparisons still require matching validation targets, training budgets and schedule evidence; document-bootstrap intervals do not measure variability across training seeds.

Validation:116 integration tests passed across sweep/search/CLI/training/evaluation/cooldown/storage; after the final source-drift guard,15 sweep/CLI tests passed. Real CPU execution covered two candidates, exact token endpoints, matching validation targets, six terminal ledger entries, hash-verified reuse, charged partial-stage rejection, lock contention, disk admission and reserved paths. No GPU sweep was launched.

## Predeclared validation comparisons

`--reference-candidate NAME` selects the reference before execution; the default is the first candidate in the declared list. The journal freezes this choice and the comparison seed. Each other candidate gets a separate verified `comparisons/NAME/comparison.json`, and the summary includes its CE difference, interval and training comparison. Re-running a completed sweep verifies these artifacts instead of recomputing them.

The sign is **candidate CE minus reference CE**, so a negative value favors the candidate on these validation targets. For document loss sums `L_candidate,i`, `L_reference,i` and common target counts `n_i`, the difference is `sum_i(L_candidate,i - L_reference,i) / sum_i(n_i)`. The paired bootstrap draws the same document indices for both models and recomputes this ratio. It does not average document mean losses equally when document lengths differ.

Intervals describe nominal document uncertainty. They do not account for training-seed variability or selection across the candidate grid. The same numeric training seed at different widths does not imply paired training randomness. Training descriptors expose differences in width/depth scaling, residual scaling, optimizer settings, warmup, schedules and token budgets; missing legacy evidence stays unknown. No comparison automatically promotes or ranks candidates as a fair experiment.

Paired-comparison integration validation:59 CPU tests passed in17.16s, including sweep execution/reuse, CLI forwarding, checkpoint-derived training descriptors and exact cooldown completion.
