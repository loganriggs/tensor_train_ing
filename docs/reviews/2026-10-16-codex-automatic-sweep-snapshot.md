# Automatic source snapshots for new sweeps

Use `sweep --snapshot-sources` to freeze a fresh experiment before launching it:

```bash
python -m tnlm.cli sweep --config configs/base.yaml --candidates grid.yaml \
  --out runs/new-sweep --max-seconds 3600 --evaluation-max-seconds 300 \
  --snapshot-sources
```

The default snapshot directory is `runs/new-sweep.sources`. An explicit directory is accepted after the flag. The snapshot contains the package's source/resources and exact config/grid bytes, with SHA256 inventory and input mappings in `manifest.json`. Existing snapshots are never overwritten. The snapshot and sweep output must be separate, non-nested directories.

The command starts a child Python process with isolated imports and bytecode writing disabled, using the verified snapshot package and captured config/grid files. It keeps the original working directory so relative data, ledger and output conventions remain intact. The child receives every current sweep control, including pruning, comparison reference and cooldown selection; it does not receive the snapshot flag. Its exit status is returned directly. The parent does not launch training from its already-imported development modules.

If all resolved candidates are dry runs, normal planning runs in the parent and reports the proposed snapshot path without creating any files. Mixed or non-Boolean dry-run policies reject. Failed child validation can leave a valid source snapshot; this does not imply that a training stage launched or completed.

The automatic flag is for fresh output directories. For completed-stage re-entry, use the original snapshot explicitly:

```bash
python -m tnlm.source_snapshot run --snapshot runs/new-sweep.sources -- \
  sweep --config configs/base.yaml --candidates grid.yaml \
  --out runs/new-sweep --max-seconds 3600 --evaluation-max-seconds 300
```

New automatic snapshots record the launch working directory and reuse it by default, including when the later command is issued elsewhere. Use `--workdir` to explicitly override it, and repeat all original sweep controls. Legacy snapshots without this optional manifest field retain the source-root default. The frozen sweep journal still validates identity and artifacts. Incomplete charged stages still fail closed; this does not add automatic recovery or adopt a drifted run. The legacy `codex_tools/source_snapshot.py` entrypoint and import API remain compatible with the authoritative packaged `tnlm.source_snapshot` implementation.

Dependencies, driver/compiler versions, datasets and external files are not bundled. Existing data-manifest/checkpoint/runtime checks still apply. Snapshot copying and parent launch orchestration are outside stage billing timers. This feature neither reserves the GPU nor changes an existing experiment. In particular, it does not modify the confirmation run or its snapshot.

Validation: **48 tests passed in10.61seconds**, including the existing21snapshot tests, legacy direct-script/import compatibility, CLI dry-run/control/rejection cases, existing sweep integration and a real isolated-child CPU train/evaluate sweep. The child journal source hashes match the copied package inventory, captured config bytes match the original input, and both training/evaluation ledger entries finish complete. Independent read-only review found no material blocker. No GPU experiment or confirmation-directory access was used.

Working-directory follow-up:37snapshot/CLI tests passed in4.81seconds, including re-entry from a different shell directory, relative input/output resolution, explicit override and legacy fallback. The real isolated-child CPU sweep was then strengthened to launch outside the package source root and passed in3.59seconds, verifying the recorded launch directory. Existing snapshot manifests were not rewritten.

## Config-edit near-miss follow-up (12:43 review)

Changing directory into a copied package and running `python -m tnlm.cli ... --config /live/base.yaml` freezes imports only. The config remains live. A shell loop that launches one candidate per invocation will read that live file again on each invocation. This was the path used by the confirmation driver; it bypassed the snapshot launcher described above.

For a new sweep, use the automatic `--snapshot-sources` command above, with the complete candidate grid. Add `--prune-completed-intermediates` when bounded checkpoint storage is needed. The entire config and candidate grid are copied before launching, and the isolated child receives their captured paths. Later edits to either original file cannot change that sweep.

For an external driver requiring separate invocations, create one snapshot **before any candidate starts**, supplying the base config and **every** candidate file via repeated `--input` options. Launch each invocation through `python -m tnlm.source_snapshot run --snapshot ... -- sweep ...`, rather than changing into the snapshot and calling its CLI directly. The launcher rewrites `--config` and `--candidates` to captured inputs, verifies their hashes, and rejects inputs that were not captured. Capturing a fresh config per candidate would still permit drift between candidates. If a driver reads an order file, freeze that file too and make the driver read its captured copy.

Existing active snapshots and confirmation files are unchanged. Moving the currently running driver onto this path requires a separately planned fresh launch; this documentation does not recommend editing its snapshot in place.

Validation of this follow-up:60focused snapshot/determinism/legacy-compatibility tests passed with CUDA hidden, including a real isolated CPU sweep whose original config and candidate files are corrupted immediately after capture. The child completes with the captured inputs and records deterministic=true loaded from YAML.
