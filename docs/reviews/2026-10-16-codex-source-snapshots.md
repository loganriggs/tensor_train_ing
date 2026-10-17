# Stable sources for confirmation runs

Use `codex_tools/source_snapshot.py` to freeze the Python package and explicitly selected configuration files before a new run. It lives outside `tnlm`, so work on the utility does not change the runtime hashes used by an already running sweep.

```bash
/venv/main/bin/python codex_tools/source_snapshot.py create \
  --source /workspace/tensor_train_ing \
  --out /workspace/tensor_train_ing/runs/source_snapshots/NEW_NAME \
  --input configs/confirmation/base.yaml \
  --input configs/confirmation/one_w512_lr0.00045_s101.yaml

/venv/main/bin/python codex_tools/source_snapshot.py verify \
  runs/source_snapshots/NEW_NAME

/venv/main/bin/python codex_tools/source_snapshot.py run \
  --snapshot runs/source_snapshots/NEW_NAME -- \
  sweep --config configs/confirmation/base.yaml \
  --candidates configs/confirmation/one_w512_lr0.00045_s101.yaml \
  --out runs/confirmation/NEW_RUN \
  --max-seconds 3600 --evaluation-max-seconds 1200 --device cuda
```

Creation exclusively claims a new destination and copies the entire package except bytecode caches, plus each explicit input. Source bytes and inventory are rechecked after copying; a concurrent change rejects creation and removes only this invocation's incomplete destination. The manifest is the completion marker. Files and directories receive read-only permissions; this discourages accidental edits but is not a security boundary against a privileged writer.

Verification checks every file hash/size and inventory, rejecting added files and symlinks. Launch uses a new Python process with `-I -B`, inserts the verified snapshot ahead of installed packages, and runs `tnlm.cli`. This prevents the mutable working tree or PYTHONPATH from selecting the training package and prevents bytecode writes into the snapshot. Full `--config` and `--candidates` options, including equals forms, are rewritten to captured copies. Uncaptured inputs and abbreviated forms reject before execution.

The working directory defaults to the original source root. Thus relative data paths, checkpoint paths, outputs and the shared spend ledger retain their meanings. An explicit `--workdir` is supported but must be outside the snapshot; choosing it changes all relative runtime paths. This utility does not copy datasets, checkpoints, dependency environments, tests or scratch scripts, and does not reserve GPUs or disk space for a run. Training/evaluation retain their existing data integrity and budget checks. It does not provide a frozen `check_math` test suite. Dependencies remain those of the selected Python interpreter, recorded by checkpoints.

A code snapshot cannot recover the exact bytes of a previous run from their hashes alone. Do not rewrite a failed sweep's pinned identity to match current code. Preserve its charged artifacts; recovery requires an explicit checkpoint branch with documented source provenance or the original source snapshot. This utility does not restart, adopt or alter an existing run.

An initial verified snapshot is published at `runs/source_snapshots/codex_tokenizer_v1`, containing 36 package files and all 15 current confirmation YAML files. It captures the tokenizer check implementation; it is **not** the source snapshot of the already running first confirmation. A CPU-only `validate_config` command through its isolated launcher verifies the actual package/config path without launching training.

Validation: 21 independent tests pass, including source mutation after destination creation and cleanup of the incomplete snapshot. Actual isolated CPU config validation evidence is in `runs/codex_source_snapshot_audit/validation.json`, bound to the published snapshot manifest SHA256. No live package changes or GPU jobs were made for this utility.
