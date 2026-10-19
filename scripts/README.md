# scripts/: experiment operations

Each tool's docstring or header has full usage. Typical flow for a new experiment `configs/EXP/{base,candidates}.yaml`:

| Step | Tool | Purpose |
| --- | --- | --- |
| 1 | `check_experiment_configs.py EXP [--keep-count K]` | Preflight: screening retention, duplicate names, peak disk vs free |
| 2 | `run_queue.sh WAIT_PID EXP[:MIN_TOK_S[:MAX_S]] ... \| path.sh` | Run experiments back to back after an optional running job; stops at the first failure |
| 2 | `run_queue_file.sh WAIT_PID QUEUE_FILE` | Same, but entries live in a file re-read before each one (reorder a running queue by editing `runs/queue.txt`) |
| 2 | `codex_msg.sh "text"` | Message Codex in tmux and verify it was submitted (retries Enter; warns if it stays in the input box) |
| | `run_candidates_v2.sh EXP MIN_TOK_S "KEEP" [MAX_S] [DEVICE]` | One sweep, one candidate at a time, in bounded disk (called by `run_queue.sh`) |
| 1b | `norm_canary.py OUT [--norms ...] [--layers] [--d-model]` | CPU end-to-end of norm packages via `tnlm train`, before GPU time |
| 1c | `poly_init_scan.py [--set k=v ...]` | Init-time per-token scale scan of a polynomial (lagged) model on real text |
| 3 | `stability_watch.sh [SEEN]` | Background alert on an abrupt attention-score blow-up in the live run |
| | `stability_trace.py metrics.jsonl ...` | Per-log stability table with flags (jump, steady runaway, absolute) |
| 4 | `make_pairs.py OUT.yaml NAME=LEFT_RUN,RIGHT_RUN ...` | Paired-comparison spec with verified absolute paths |
| | `print_pairs.py comparisons.json` | One line per pair: ΔCE and document-bootstrap CI |
| | `plot_scaling_losses.py` | Training-loss figures for docs/ (Codex) |
| 4a | `analyze_kept_finals.sh SWEEP TAG CAND...` | CPU `tnlm analyze` of a sweep's kept finals with recorded configs, then the cost table |
| 4b | `norm_cost_table.py LABEL=analysis.json ...` | Analysis-cost table per normalization package (5 metrics) |
| 4c | `find_orphan_states.py [--keep ...]` | List checkpoints whose run already has an evaluation (deletion candidates; deletes nothing) |
| 4d | `fit_lr_optimum.py [--pattern ...]` | Per-group LR optimum (quadratic in log LR) and power law of the optimum across groups |
| 4d | `throughput.py RUN_DIR [...]` | Steady-state tokens/s per run (second half of metrics.jsonl); catches throughput regressions |
| 4e | `scaling_grid.py SIZE [--lr-mult M] [--tpp 5,10,20,40] [--plan]` | Scaling-law grid: one constant-LR parent per size, cooled branch + evaluation per token budget; resumable, prunes states |
| 4e | `fit_scaling_law.py [--params total\|nonemb] [--compute C ...]` | Chinchilla fit L = E + A/N^α + B/D^β of the grid evaluations (Huber/L-BFGS, torch); compute-optimal N, D, tokens/param |
| 4d | `find_runs.py KEY=VALUE ... [--top N]` | Evaluated runs matching training-descriptor filters, best CE first (like-for-like comparators) |
| 5 | `export_results.py` | Weight-free mirror of runs/ into git-tracked results/ |
| | `autocommit_results.sh` | Hourly export, commit and push of results/ (runs in the background) |

Rules: never edit a script, config or candidates file that a running job is using; derive paths from runs rather than writing them by hand.
