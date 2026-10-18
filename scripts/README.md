# scripts/: experiment operations

Each tool's docstring or header has full usage. Typical flow for a new experiment `configs/EXP/{base,candidates}.yaml`:

| Step | Tool | Purpose |
| --- | --- | --- |
| 1 | `check_experiment_configs.py EXP [--keep-count K]` | Preflight: screening retention, duplicate names, peak disk vs free |
| 2 | `run_queue.sh WAIT_PID EXP[:MIN_TOK_S[:MAX_S]] ... \| path.sh` | Run experiments back to back after an optional running job; stops at the first failure |
| | `run_candidates_v2.sh EXP MIN_TOK_S "KEEP" [MAX_S] [DEVICE]` | One sweep, one candidate at a time, in bounded disk (called by `run_queue.sh`) |
| 1b | `norm_canary.py OUT [--norms ...] [--layers] [--d-model]` | CPU end-to-end of norm packages via `tnlm train`, before GPU time |
| 3 | `stability_watch.sh [SEEN]` | Background alert on an abrupt attention-score blow-up in the live run |
| | `stability_trace.py metrics.jsonl ...` | Per-log stability table with flags (jump, steady runaway, absolute) |
| 4 | `make_pairs.py OUT.yaml NAME=LEFT_RUN,RIGHT_RUN ...` | Paired-comparison spec with verified absolute paths |
| | `print_pairs.py comparisons.json` | One line per pair: ΔCE and document-bootstrap CI |
| | `plot_scaling_losses.py` | Training-loss figures for docs/ (Codex) |
| 5 | `export_results.py` | Weight-free mirror of runs/ into git-tracked results/ |
| | `autocommit_results.sh` | Hourly export, commit and push of results/ (runs in the background) |

Rules: never edit a script, config or candidates file that a running job is using; derive paths from runs rather than writing them by hand.
