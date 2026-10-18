# Compact, shared sampler index

PackedSampler now reads document boundaries from a compact, read-only NumPy
memory map and loads full JSON metadata only for the current batch. It no longer
retains Python document dictionaries, duplicate-ID sets, or permutation lists
proportional to the corpus size. RankPackedSampler uses the same index.

## Measured 3B result

Fresh CPU processes on the existing Ryzen 9 5900X host, train split, batch 16,
context 512, seed 0. The accepted dataset contains 3,009,867,881 training tokens
and 4,287,242 training documents. All times include artifact checksum verification;
Python/Torch import time is excluded, but their memory is included in peak RSS.
No GPU work was launched. GPU training elsewhere on this host continued.

| Implementation | Startup through first batch | Peak process RSS |
| --- | ---: | ---: |
| Frozen legacy sampler | 65.03 s | 11.625 GB |
| Compact index, first shuffle/row-count build | 16.80 s | 0.602 GB |
| Compact index, reused shuffle/row-count cache | 9.80 s | 0.554 GB |

The cached run reduces peak RSS by 95.2% and startup time by 84.9%. All four
measured batch hashes, lineage hashes, and final sampler state match the legacy
implementation exactly. The earlier 59.15 s / 11.62 GB smoke measurement is
consistent with this fresh baseline; it is not substituted for the baseline.
These are local filesystem-cache measurements, not cold-disk or H100 results.
Final compact measurements use hard links to the identical accepted dataset and
index bytes in an isolated benchmark directory, with fresh packing caches; this
avoids deleting caches another process might be using.

The separate one-time metadata-index build took 177.03 s and 0.554 GB peak RSS.
Its four split payloads total 176,608,800 bytes (40 bytes per document); the
training permutation adds 34,297,936 payload bytes plus its NumPy header.
Small CPU tests overlapped part of index construction and may have overlapped
the first-use measurement. The cached startup measurement ran separately. Index construction is now part of preparation, before
manifest publication; the older preparation-throughput report predates this step.

Raw measurements are in [benchmark_sampler_3b](benchmark_sampler_3b/).
The accepted manifest remains unchanged:
`390b37a6c2fd0e3c51e028ef0d3719f91b690c3b72c4cbd220ed5f100e886543`.

A near-end seek plus two checkpoint restores and exact replay took **0.058 s**
after startup, with **0.592 GB** process peak RSS. Before adding sparse packing
checkpoints, the same compact-index operation took 129.58 s at 0.630 GB. The saved
position and resulting batch hash match exactly across both implementations.

## Storage and replay

Preparation writes `.sampler-index-v1/` beside the original artifacts. Existing
format-1 datasets build it once on first access, under a filesystem lock. The
index is derived, checksummed, and bound to the original manifest and document
metadata hashes. It does not change manifest bytes, the packing version, or
checkpoint identity. Original artifact verification remains enabled.

Each document has five 64-bit fields: token start/end, numeric ID, and JSON byte
offset/length. Full metadata uses positional reads. Duplicate IDs and metadata
coverage are validated during construction with disk-backed SQLite, replacing
unbounded Python sets. Verification uses bounded chunks. Publication uses fsync,
locking, and atomic replacement. Corrupt or stale document indexes fail explicitly;
`ensure_sampler_index(..., rebuild=True)` rebuilds from the verified source.

Permutation caches use the same Python `random.Random` shuffle as the legacy
sampler, stored as read-only mapped int64 arrays. They are keyed by manifest,
split, seed and epoch. Context-specific row counts and sparse packing checkpoints every 4,096 documents
are cached separately. Resume prefix validation scans at most 4,096 document
lengths; row seeks and context forks jump to a nearby sparse checkpoint. All
ranks map the same files; page-cache storage is shared. Disposable packing caches
are regenerated if their checksums or identities fail validation. First-use row
counting uses one compact int64 lengths array and bounded file reads, avoiding
faulting the entire structured index into each process.

On Linux, indexed accesses periodically release process-local mmap page mappings
with MADV_DONTNEED. This leaves file contents and the shared filesystem cache
intact, trading minor page faults for limited residency during long scans. The
release interval is 1,024 indexed accesses; actual RSS depends on OS fault-around
behavior. On platforms without madvise, correctness is unchanged but this
residency control is unavailable. Direct access to exposed NumPy views bypasses
access accounting; the sampler uses the accounted accessors.

## Scope of the readiness result

This verifies the actual 3B corpus, not a 30B or eight-H100 run. At the observed
training-document density, 30B tokens imply about 42.7 million training documents:
approximately 1.71 GB of shared index files and 0.342 GB per cached permutation.
The first cache builder temporarily holds roughly 16 bytes/document for order
and lengths (about 0.684 GB at that density), plus the Python/Torch process and
bounded buffers. Other ranks reuse the mapped cache rather than retaining those
construction arrays. These sizes are projections, not a measured 30B RSS claim.

Tests compare a frozen legacy sampler against the new path across contexts,
seeds, splits, epochs, partial batches, checkpoint replay, context forks and rank
slicing. They also exercise concurrent construction, corruption, metadata
validation, page release, and byte-identical serial/parallel preparation including
the derived index. The final broad CPU run passed **1,073 tests** (4 warnings, 101.46 seconds),
including distributed replay and all 46 compact-index tests. GPU-only kernel and
parallel suites were excluded; CUDA was hidden.

Reproduce startup with `scripts/benchmark_sampler.py --manifest MANIFEST --output
REPORT.json`. `--legacy-source` accepts the frozen `tnlm/data.py` from commit
`218588f`; `--build-index-only` measures construction separately. Run a second
fresh process to measure reuse. `--verify-long-resume` additionally scans near the
end of the epoch and checks restored batch and lineage equality.
