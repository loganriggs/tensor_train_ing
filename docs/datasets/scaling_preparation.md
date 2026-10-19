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
The already-running frozen snapshot uses an external supervised guard with the
same thresholds; the snapshot itself remains unchanged.

Future launches set the preparation worker to nice 10 before it spawns tokenizers
(and preserve an already lower priority). The disk-monitoring coordinator keeps
its original priority. The current frozen run and its existing children were
reniced to 10 in place at the 18:45 request; its snapshot was not modified.

The release pipeline uses a frozen code snapshot and supervisor. Preparation
success alone is not acceptance: `scripts/audit_scaling_data.py` independently
checks every artifact hash, int32 payload sizes, metadata ranges, minimum train
tokens, reference policies, held-out overlap across all splits, and unchanged
historical validation identities. A sampler benchmark then checks ordinary batch
construction and near-end checkpoint replay. Logs and detailed measurements live
under `runs/codex_data_13b/`; final release evidence is copied into this directory
when the audit finishes.

Validation of the implementation: 1,153 fast CPU tests passed, including
serial/parallel byte identity and legacy sampler behavior. Additional focused
checks exercise disk reclamation, fixed-validation exclusions, explicit raw-file
pruning, audit failures, and a real two-worker preparation. No GPU is used by this
pipeline.

The pause policy also has an actual CPU process-group test: parent and child
progress stop under SIGSTOP and continue under SIGCONT without resetting. Eleven
focused launcher tests passed after the disk-policy update.
