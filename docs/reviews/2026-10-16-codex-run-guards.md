# Throughput canaries and explicit ledger reconciliation

## Sweep throughput gate

Fresh sweeps accept `--canary-steps N --min-tok-s X`, optionally `--canary-warmup-steps W` (default12). `--expect-tok-s` is an alias for the minimum threshold, not an expected mean or tolerance band. Controls are frozen in the sweep identity and forwarded into automatic source snapshots.

The gate measures the first N completed parent updates after the first W updates. This is an explicit timing warmup, independent of the LR warmup. Recompilation after W remains timed. Throughput is summed actual prediction targets divided by summed synchronized step seconds, including sampling, preparation, transfers, accumulation, backward and optimizer work. Checkpoint/metric writes and canary evidence publication are outside those step timers. The whole stage remains billed.

Passing continues the same parent trajectory; no canary-only training segment or repeated optimizer updates are introduced. Low throughput or an incomplete sample fails the stage and prevents cooldown/evaluation or automatic retry. Completed metrics and the last recoverable checkpoint remain; a failed canary does not produce a successful final checkpoint. A parent horizon too short for W+N is rejected before launching. Choose the threshold for the actual shape, precision, backend, context and accumulation workload; the measured67–68k targets/s is not a threshold for every model.

Evidence is written atomically to `candidate/parent/canary.json`, with controls, selected steps, target counts, durations and outcome. Completed sweep-stage inventories bind that file to the frozen source/config identity. A passed gate does not establish future throughput, model quality or full-run completion. Dry planning launches nothing and creates no canary artifact.

## Reconcile one identified terminal run

Inspect explicitly supplied evidence first:

```bash
python -m tnlm.cli reconcile --ledger PATH --run-id EXACT_RUN_ID --metrics METRICS_JSONL
```

This reports the current row SHA256 and observed elapsed-cost lower bound without rewriting the ledger. Metrics do not prove termination. Evidence without embedded run identity is labeled unverified; it requires operator attribution. Incomplete JSON lines reject rather than silently guessing.

Apply only after establishing that this exact job and all ranks/workers have stopped:

```bash
python -m tnlm.cli reconcile --ledger PATH --run-id EXACT_RUN_ID \
  --metrics METRICS_JSONL --apply --expected-row-sha256 INSPECTED_SHA \
  --operator NAME --reason 'identified job terminated' --all-workers-stopped
```

Without a known total, the status becomes `reconciled_unknown_tail`. Observed spend is recorded while enough reservation remains to preserve at least the prior allocation. A later operation can use a fresh row digest and an authoritative `--total-elapsed-seconds` or `--total-cost-usd` to release the reservation and set `reconciled_terminal`. Totals below observed bounds, conflicting totals, stale digests and already-final entries reject. Overruns remain visible rather than being discarded by admission limits.

Every reconciliation preserves the old row/digest, operator/reason/attestation, evidence hashes and supplied amounts. It never marks a killed training stage complete or permits sweep auto-retry. Neither API nor CLI infers death from old timestamps or missing PIDs. No existing confirmation ledger was inspected or modified while implementing this feature.

Final integrated gate after CLI/NCCL setup, reconciliation and canary integration: **772CPU tests passed,3warnings,64.02seconds**, xdist8, CUDA hidden, kernel/parallel files excluded. No new CUDA performance or replay claim follows from this gate.
