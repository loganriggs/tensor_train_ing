# Context continuation: sampler contract and trainer integration

Brief §6 requires short continued-training branches at 2K, alongside matched
early-from-scratch training. Ordinary resume pins context. An explicit context
fork permits a strictly larger `data.context`, preserving the other mathematical
and data settings and the existing rules for token-horizon promotion.

Set `data.context: 2048` in a copy of the parent configuration, then run:

```bash
tnlm train --config context_2k.yaml --run-dir NEW_BRANCH --resume PARENT_CHECKPOINT \
  --context-fork --max-steps 100 --max-seconds 600 --device cuda
```

For a distributed parent, use the same world size and checkpoint layout under
`torchrun`, adding `--distributed`. The option also works with explicit cooldown
arguments. To resume the resulting branch at its current context, use ordinary
`--resume` without `--context-fork`; this includes its initial checkpoint while
the transition mask is still pending. A dry run reports the requested context
and parent with validation pending, without opening a placeholder parent.

The global LR schedule, warmup, optimizer moments, RNG, normalization buffers and
token counters are restored. Lagged norms are not recalibrated. A constant
schedule retains its existing upward token-promotion rule; cosine retains its
exact original horizon, so select a parent with remaining updates. Adding
cooldown explicitly selects the existing separate cooldown schedule as well as
the context intervention. Both trainers validate configuration compatibility
before spend admission and use the restored cursor for checkpoint storage plans.

`sampler.fork_context(context)` returns a separate sampler sharing the immutable
token cache. It preserves the consumed source-stream watermark, finds the
containing row in the new canonical packing, and retains earlier tokens as input
prefix while masking their already-consumed targets. The original sampler and
its last-batch mappings remain unchanged. Canonical row seeking is proportional
to document count, without walking all earlier rows or materializing token data.

At a mid-document boundary the saved token cursor names the overlap input token,
which was already a target in the preceding row. The first new target has offset
`token_cursor + 1`. At a document boundary the first predictable target is offset
1. An exhausted epoch advances to the next deterministic permutation. Numeric
document IDs do not determine traversal order.

A versioned pending transition records the validated original source state and
context. Repeated forks before consuming the transition preserve that original
watermark. Loading validates both the source cursor and the derived destination
cursor before modifying the live sampler. Distributed samplers fork the common
global stream before rank slicing; ranks skipping the first global row also
consume its transition state.

This conserves the **consumed sampler stream**. Existing trainers discard excess
targets when an update quota ends inside a batch; checkpoints retain aggregate
discard counts, not those pending targets. A context fork cannot recover that
earlier discarded history. Replayed prefix inputs can also affect normalization
statistics and input exposure even though their targets are masked. Both effects
must remain explicit in experimental interpretation.

Checkpoints retain `progress.context_history`: context, starting scored-token and
step counters, parent identity and source sampler watermark for each intervention.
Fresh training starts with an explicit context segment at zero. Legacy parents
without history remain unknown before the recorded fork. Evaluation and analysis
descriptors project this into context schedules; comparisons cannot treat two
partial histories as matched. JSON/CSV training summaries retain this distinction.
Thus a 512-to-2K branch is not labeled as having trained entirely at 2K.

`progress.context_fork_exposure` accumulates known transition-mask counts:
`masked_prediction_targets` and `prefix_tokens_at_or_before_watermark`. The latter
includes the overlap token, which may previously have appeared only as a target,
so it is not a count solely of repeated input positions. Padding and lookahead
slots are excluded. These counters are separate from quota-discard counts and
ordinary input-token exposure. Distributed ranks sum their local counts and
agree on the global sampler state before training begins.

Runtime RoPE supports a changed context without parameter-shape changes. Available
prefixes and row-local positions change during repacking; preserving the source
watermark does not claim identical inputs or losses across contexts. Empirical
2K capability and interpretability comparisons remain separate required evidence.

Validation: 60 data/rank/context tests passed, including independently enumerated
target identities through epoch boundaries, pending serial/rank replay, repeated
forks, malformed transitions and a far-offset seek without row iteration. Root's
additional audit compared 3,690 cross-epoch streams over four seeds and multiple
old/new contexts. The full 528-test CPU suite passed in 137.20 seconds, including
owned distributed gates; CUDA kernel/parallel test files were excluded.

Trainer integration validation: **556 CPU tests passed in 153.09 seconds**,
including all owned distributed gates (three warnings, no skips; CUDA
kernel/parallel test files excluded). Serial AdamW/Muon and two-rank CPU/gloo
AdamW/lagged plus Muon/spectral branches replay exactly from split and pending
initial checkpoints. Initial fork checkpoints preserve parameters, all norm
buffers, optimizer state and RNG. Tests cover context history, global transition
exposure, prebilling rejection, CLI dry runs, fixed-horizon cosine and exact
cooldown endpoints. This does not establish CUDA/NCCL replay or empirical 2K
quality; those experiments remain outstanding.
