# Confirmation planning and exact cooldown endpoints

`plan_search --config BASE.yaml --candidates CANDIDATES.yaml` resolves an ordered, explicit list without launching jobs or changing the spend ledger. Candidate files contain:

```yaml
- name: lr-low
  overrides:
    train:
      lr: 0.00015
- name: lr-mid
  overrides:
    train:
      lr: 0.00045
```

Every entry includes its full resolved configuration, configuration SHA256, and ordinary single-trajectory plan. The result sums projected checkpoint storage and costs. If a candidate lacks measured throughput or a rate, its cost remains unknown and the total is null. Estimates cover fresh uncooled full rungs; separate cooldown and evaluation work must be budgeted additionally. The storage sum is conservative and advisory, with no reservation or automatic deletion. This adds reproducible grid planning; a resumable execution driver remains separate work.

For an exact cooldown endpoint, use `cooldown --cooldown-steps K --cooldown-end-tokens D` with the existing explicit parent checkpoint. Compute K from the **actual saved parent token count** P: `K = ceil((D-P)/B)`, where B is prediction tokens per update. The final update uses the remaining targets. Omitting the endpoint preserves the existing step-only behavior.

For D=50,000,000 and B=8,192, a parent checkpoint first crossing90% lands at P=45,006,848. K=610 and the final update contains4,224 targets. Using611steps based on rounding10% of the nominal rung instead would end at50,012,160. The endpoint is pinned in checkpoint progress and must remain consistent on resume. Evaluation completion uses the checkpoint endpoint rather than assuming every update is full-sized.

The linear schedule retains its existing convention: update index j=0,...,K-1 uses multiplier1-j/K. The last executed update has multiplier1/K; zero is the next schedule boundary. This change makes token exposure exact, without changing that schedule convention or claiming identical floating-point updates across hardware.

Validation:468 CPU tests passed (3warnings,118.82s), including two-rank exact-endpoint cooldown replay. CLI argument forwarding also passed. GPU kernels and Claude parallel tests were excluded; no CUDA/NCCL claim.
