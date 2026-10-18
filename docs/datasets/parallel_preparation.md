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
