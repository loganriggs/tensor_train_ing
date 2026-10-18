# Preparing large frozen datasets

`tnlm prepare_data` supports `--workers N` and
`--tokenization-batch-bytes BYTES` (defaults: 1 and 1048576). These are execution
controls: changing them preserves the manifest and all dataset artifact bytes.
Parallel workers use the pinned GPT2 tokenizer; injected test tokenizers use the
serial path.

```bash
python -m tnlm.cli prepare_data \
  --hf-dataset HuggingFaceFW/fineweb --revision FULL_COMMIT \
  --hf-data-files EXPLICIT_SHARDS \
  --hf-shard-checksums checksums.json --hf-local-shards local_paths.json \
  --out data/fineweb_large --max-tokens 3100000000 \
  --max-documents 10000000 --workers 8 \
  --exclude-validation-manifest data/existing_validation/manifest.json
```

Raw shard bytes must match the declared SHA256 values. Source scanning and
selection remain serial; tokenization runs across processes. Selection uses a
SQLite index with a 32 MiB cache, periodic commits and disk-backed temporary
storage. Document metadata is written as each document is tokenized. Workers
consume canonical rank order through at most twice the worker count of queued
batches. Each batch holds at most 128 documents and the requested UTF-8 byte
budget, except that an oversized individual document runs alone. Thus memory
also depends on the largest source document and tokenizer/Parquet buffers, not
just the batch setting. This does not impose a hard process RSS limit.

The output remains format version 1, with little-endian **int32** token files.
Budget **12 GB for 3B tokens** or **120 GB for 30B tokens**, plus metadata, raw
Parquet files, SQLite selection scratch and filesystem headroom. Source bounds
and token caps do not limit the raw source scan. All declared shards are read and
hashed; duplicated IDs with conflicting contents still fail preparation.

`--exclude-validation-manifest` verifies the reference manifest and document
metadata checksums, then excludes all source documents matching a reference val
ID, original content hash or CRLF-normalized content hash. Exclusion runs before
selection, affects all output splits, and is recorded in `heldout_exclusion`.
Reference train/test/calibration rows do not become exclusions. Duplicate-ID
conflicts still fail even when the documents would be excluded. The reference
identities live in indexed SQLite tables rather than an unbounded Python set.

Hash-based split membership is stable as the source grows. It does not prevent
identical content under different IDs from crossing split boundaries. Before
training, audit the accepted fixed validation set against training content.
The benchmark audits every existing validation document against the new
train/calibration/test documents by ID and raw/processed full-document hashes;
this does not detect near duplicates. Continue using the existing validation
manifest for comparisons instead of silently adopting the expanded set's val
split.

`scripts/benchmark_prepare_data.py` measures the complete preparation, including
source hashing, selection, tokenization, artifact hashing and fsync. Raw acquisition
and the subsequent heldout audit are separate. It samples summed parent/descendant
RSS every 0.1 seconds; shared pages count once per process and short peaks may be
missed. The source specification lists explicit dataset/revision/shard hashes and
local files, making acquisition independent of the timed run. The benchmark
enables validation exclusion and returns nonzero if its independent audit finds
any remaining overlap.

The downstream `PackedSampler` still loads document metadata into memory. The
preparation memory bounds do not establish training-loader memory bounds for a
30B-token corpus, nor do they validate multi-GPU training.

## Measured 3B verification (2026-10-18)

Implementation commit: `fa8c817`. Accepted manifest:
`data/fineweb_3b_heldout/manifest.json`, SHA256
`390b37a6c2fd0e3c51e028ef0d3719f91b690c3b72c4cbd220ed5f100e886543`.
Keep `data/fineweb_confirmation_320m/manifest.json` as the validation manifest.

| Measurement | Result |
| --- | ---: |
| Total output tokens | 3,100,000,000 |
| Training tokens | 3,009,867,881 |
| Output documents | 4,415,220 |
| Whole preparation elapsed | 1,140.71 s (19.01 min) |
| Whole preparation throughput | 2,717,608 tokens/s |
| Training-token throughput | 2,638,594 tokens/s |
| Sampled peak aggregate process RSS | 5,883,887,616 bytes (5.88 GB) |
| Sampled peak parent RSS | 2,947,469,312 bytes (2.95 GB) |
| Tokenizer workers | 8 |
| Fixed validation documents audited | 4,496 (includes the 1,476 evaluation documents) |
| Heldout matches in train/calibration/test | 0 / 0 / 0 |

Hardware: AMD Ryzen 9 5900X, 24 logical CPUs available. Existing GPU training
continued; no GPU was used for preparation. The 986-test CPU gate overlapped the
early source scan, so these are observed shared-node results, not an isolated
maximum-throughput claim. Downloads and the post-preparation audit are excluded
from elapsed time; source hashing, selection, tokenization and durable artifact
publication are included. Peak RSS is a 0.1-second sampled sum, with shared pages
counted once per process.

The initial unfiltered run found three cross-ID exact-content validation matches
in training and was rejected and deleted. The accepted rerun excluded 4,500
source identities (the 4,496 reference documents plus four content aliases in the
full declared source) before canonical selection. The independent final audit
checked all 4,496 validation documents, not just the 1,476 evaluation subset.
This is exact ID/raw/normalized-content exclusion, not near-duplicate detection.

The production sampler successfully loaded the accepted data, constructed a
16×512 batch and read token offsets above 2^31. Its separate peak RSS was
11.62 GB and startup plus first batch took 59.15 seconds. **This exposes a
remaining H100 readiness issue:** the current sampler materializes document
metadata per process. A 30B corpus and multiple ranks require separate loader
memory work/measurement; the 5.88 GB preparation result does not cover training.

Evidence: [benchmark](benchmark_3b/benchmark.json),
[source specification](benchmark_3b/source.json),
[manifest](benchmark_3b/manifest.json),
[rejected overlap examples](benchmark_3b/rejected_overlap_evidence.json),
[sampler smoke](benchmark_3b/sampler_smoke.json). The benchmark records exact
implementation hashes, package versions and thread settings. Serial/parallel
byte identity is covered by fixtures and a legacy implementation oracle; the
large accepted corpus was prepared with eight workers. No 30B throughput claim
is made from this 3B measurement.
