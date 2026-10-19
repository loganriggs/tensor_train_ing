# FineWeb preparation for the sub-1B scaling grid

The scaling corpus uses the same pinned FineWeb revision, GPT2 tokenizer,
normalization, stable-ID deduplication and fixed held-out exclusion as the accepted
3B corpus. Evaluation must continue to use
`data/fineweb_confirmation_320m/manifest.json`; the new preparation's random
validation split is not the historical evaluation set. All 4,496 documents in
that fixed validation split are excluded by ID and exact raw/normalized content,
which also protects the 1,476 documents used in the earlier evaluations.

Token storage remains little-endian int32. The disk reduction is confined to
preparation intermediates and does not change manifest format, token bytes,
selection order, split assignment or sampler replay.

The preparer stores candidate text as compressed UTF8 in SQLite. After the full
source scan has checked duplicate-ID conflicts and exclusion, it drops the
completed duplicate ledger. During tokenization it consumes the front of the
ranked index in bounded batches and progressively returns free SQLite pages to
the filesystem. This avoids retaining the entire text database beside the final
token payload. Preparation remains atomic rather than resumable: failed output
is discarded, and already-pruned raw shards must be downloaded again to retry.

`scripts/prepare_scaling_data.py` supervises the worker and samples aggregate RSS
and free disk. Its explicit `--prune-raw-root` option deletes only declared raw
parquet files inside a dedicated disposable directory, after each shard has been completely read, its post-scan checksum has passed,
and its parquet reader has closed. Existing raw files outside that directory remain untouched. The
source specification is frozen beside the report. The launcher pauses its entire worker process group below 35 GiB free and
resumes at 40 GiB, preserving progress and leaving margin over the requested
25 GB GPU-checkpoint reserve. Pauses and their durations are recorded. This is a
sampled guard, not a filesystem quota or reservation against concurrent writers.
The completed frozen run used an external supervised guard with the
same thresholds; the snapshot itself remained unchanged.

Future launches set the preparation worker to nice 10 before it spawns tokenizers
(and preserve an already lower priority). The disk-monitoring coordinator keeps
its original priority. The completed frozen run and its existing children were
reniced to 10 in place at the 18:45 request; its snapshot was not modified.

The release pipeline uses a frozen code snapshot and supervisor. Preparation
success alone is not acceptance: `scripts/audit_scaling_data.py` independently
checks every artifact hash, int32 payload sizes, metadata ranges, minimum train
tokens, reference policies, held-out overlap across all splits, and unchanged
historical validation identities. A sampler benchmark then checks ordinary batch
construction and near-end checkpoint replay. Logs and detailed measurements live
under `runs/codex_data_13b/`; final release evidence is retained in
[`benchmark_13b/`](benchmark_13b/release.json).

## Accepted 13B release — 2026-10-19

The accepted corpus is `data/fineweb_13b_heldout/manifest.json`, with
**12,612,675,813 train tokens**, 13,000,000,000 tokens across all splits, and
18,143,565 documents (17,604,288 training documents). Its manifest SHA256 is
`70635d120e3dd619fa8eff8f94aacf8c23183a9ed3f37d11249e04c99dec2bac`.
The existing 3B tokenized dataset remains available for active jobs.

[Release acceptance and evidence hashes](benchmark_13b/release.json) bind the
[manifest](benchmark_13b/manifest.json), [source specification](benchmark_13b/source.json),
[preparation measurements](benchmark_13b/preparation.json),
[independent audit](benchmark_13b/audit.json), and
[sampler benchmark](benchmark_13b/sampler.json). The preparation report's
`accepted: false` is preserved verbatim: it records the earlier preparation-only
stage. The release record accepts the corpus after the subsequent checks passed.

The audit verified every artifact hash, token-file length and document range,
with zero fixed-heldout matches by stable ID, raw content SHA256 or normalized
content SHA256 in every output split. The historical validation manifest is
unchanged, SHA256
`d93d1555f9808f7a6031fd4e6245fccecbe6e46d8e1307766a6f458502f9c4c5`.
Keep using `data/fineweb_confirmation_320m/manifest.json` for evaluation. This
is an exact whole-document exclusion check, not a near-duplicate or partial-text
check; the audit compares tokenizer/preprocessing provenance rather than
retokenizing all sources.

| Measurement | Observed value |
| --- | ---: |
| Preparation wall time | 12,908.59 s (3 h 35 min 9 s) |
| Total output tokens / preparation wall time | 1,007,081 tokens/s |
| Train tokens / preparation wall time | 977,076 tokens/s |
| Peak sampled aggregate process-tree RSS | 5,166,456,832 bytes (4.81 GiB) |
| Minimum sampled free disk | 43,010,117,632 bytes (40.06 GiB) |
| Disk-guard pauses | 0 |
| Independent audit time | 210.13 s |
| Sampler initialization, including artifact verification | 55.244 s |
| Sampler first batch | 0.0214 s |
| Sampler peak process RSS, including imports | 813,101,056 bytes (0.757 GiB) |
| Near-end checkpoint restore and exact batch replay | Passed, 0.0831 s |

Preparation wall time includes source scanning, tokenization and index creation;
it excludes source acquisition, the abandoned initial scan attempt, independent
audit and sampler benchmark. RSS was sampled every 0.25 seconds and sums parent
and descendant RSS, counting shared pages once per process. The worker and its
children were reniced to 10 during this run at the 18:45 request, so the result
is not a controlled comparison of priorities or an isolated throughput run.
The recorded frozen implementation hashes identify the measured code; later
reader-lifetime cleanup and the launcher's native pause/nice policy were not
retroactively applied to that snapshot.

The external guard finished without a pause and stayed above the requested
25 GB reserve in its samples; sampling does not establish a continuous disk
reservation. [Guard evidence](benchmark_13b/disk_guard.json),
[incremental raw pruning](benchmark_13b/preparation.pruning.json) and
[authorized old-raw cleanup](benchmark_13b/old_raw_cleanup.json) record space
management. The old-raw cleanup reverified and removed 6,444,262,641 bytes of
consumed source files; it did not remove the old 3B tokenized corpus. Some source
specification local paths now point to deliberately pruned files; pinned remote
paths, revision and hashes remain available for reproducibility.

The sampler check used a fresh CPU process, batch size 16, context 512 and seed 0.
Four ordinary batches have recorded hashes, and a near-end epoch-0 checkpoint
replayed its next batch exactly. This validates that sampled replay case; it
does not establish that a trainer consumes every fetched token or enforce a
single-epoch limit. Trainer token-accounting and epoch-wrap findings are separate
from acceptance of these corpus artifacts.

Validation of the implementation: 1,153 fast CPU tests passed, including
serial/parallel byte identity and legacy sampler behavior. Additional focused
checks exercise disk reclamation, fixed-validation exclusions, explicit raw-file
pruning, audit failures, and a real two-worker preparation. No GPU is used by this
pipeline.

The pause policy also has an actual CPU process-group test: parent and child
progress stop under SIGSTOP and continue under SIGCONT without resetting. Eleven
focused launcher tests passed after the disk-policy update.
