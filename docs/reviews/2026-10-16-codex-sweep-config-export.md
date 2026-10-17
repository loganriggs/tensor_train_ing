# Export a recorded sweep candidate configuration

```bash
python -m tnlm.cli sweep-config \
  --sweep runs/confirmation/v3/w512_lr0.00045_s101 \
  --candidate w512_lr0.00045_s101 \
  --out /tmp/w512-recorded.yaml
```

Omit `--out` to print YAML to stdout. The output is directly usable as `analyze --config /tmp/w512-recorded.yaml`. The command selects exactly one named `identity.trajectories` entry from `DIR/sweep.json` and verifies both the sweep identity SHA-256 and selected configuration SHA-256. Unknown or duplicate candidate names and inconsistent hashes reject before output. An existing output file is never overwritten.

Export preserves recorded configuration values and omissions, including the original data/ledger paths and dry-run policy. Current dataclass defaults are not injected, and the live base config is never read. YAML comments record the journal path, journal-byte hash, identity hash, config hash and selected candidate. This is an extraction command, not a checkpoint completion or integrity certificate; analyze still performs its normal checkpoint validation. Relative paths retain their recorded meaning and should be used from the original launch working directory.

No training, analysis, data loading, ledger writes or GPU allocation occurs. A read-only export from the completed `w512_lr0.00045_s101` confirmation journal round-tripped exactly to its recorded config. Ten focused export/planning/sweep CLI tests pass, including legacy omission preservation, stdout/file output, no overwrite and corruption/ambiguity rejection. Confirmation files were not changed.
