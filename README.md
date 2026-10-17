# tnlm: tensor-network language models

An experiment platform for bilinear-MLP / polynomial-attention language models. The plan and its
acceptance criteria are in [`tensor_network_research_brief.md`](tensor_network_research_brief.md).
Who owns which files (Claude and Codex work in parallel) is in [`COORDINATION.md`](COORDINATION.md).

## Setup
```bash
source /venv/main/bin/activate
uv pip install -e ".[dev]"      # pinned versions in pyproject.toml
pytest -n 8 -m "not slow"       # quick loop (~30 s on 24 cores); `pytest -n 8` runs everything incl. multi-process tests (~50 s)
```

## Layout
| Path | Contents |
| --- | --- |
| `tnlm/config.py` | `ModelConfig`/`TrainConfig` dataclasses; `Config.from_yaml` |
| `tnlm/model.py`, `attention.py`, `normalization.py` | the model: reference attention, six norm packages, P/N/G operator audit |
| `tnlm/kernels/` | Triton polynomial attention (fwd+bwd), chunked fused linear cross-entropy, `tune` (per-GPU tile sweep) |
| `tnlm/optim.py` | AdamW reference grouping; Muon (hidden matrices) + AdamW |
| `tnlm/parallel.py` | FSDP2 sharding and the loss-normalization contract |
| `tnlm/analysis.py` | §10 interpretation diagnostics: `analyze(model, batches, out_dir)` |
| `tnlm/shapes.py`, `coordcheck.py` | presets 100M–32B with a memory planner; width coordinate check |
| `tnlm/data.py`, `tasks.py` | immutable manifests, packing, samplers; synthetic mechanism tasks |
| `tnlm/train.py`, `distributed_train.py`, `calibration.py`, `budget.py` | training loops (1-GPU / FSDP2), lagged calibration, spend ledger |
| `tnlm/checkpoint.py`, `distributed_checkpoint.py` | atomic full-state and sharded checkpoints |
| `tnlm/evaluate.py`, `reporting.py`, `runtime.py`, `cli.py` | held-out and mechanism evaluation, summaries and plots, CLI |
| `configs/` | canonical run configs (`screen_100m.yaml` … `funded_32b.yaml`); experiment sweeps as `configs/<exp>/{base,candidates}.yaml` (e.g. `confirmation/`, `joint50m/`, `width2048/`) |
| `tests/` | test suite; `tests/dist_workers/` holds multi-process workers launched by tests |
| `scripts/` | reusable sweep tools: `run_candidates_v2.sh`, `check_experiment_configs.py`, `print_pairs.py`, `stability_trace.py`, `plot_scaling_losses.py` (tested in `tests/test_scripts.py`); tracked, unlike `runs/` |
| `docs/scaling.md` | measured throughput, memory and width-transfer results |
| `docs/reviews/` | periodic code-health and efficiency reviews |
| `scratch/` | **one-off scripts and experiments, deleted once results settle.** Nothing in `tnlm/` or `tests/` may import from here |

## Commands
`python -m tnlm.cli <cmd>` (or `tnlm <cmd>`): `validate_config`, `prepare_data`, `check_math`, `profile`,
`plan_search`, `train`, `cooldown`, `analyze`, `evaluate`, `summarize`, … Paid runs need
`project.dry_run: false` and an actual `gpu_hourly_rate_usd`.

## Running an experiment sweep
`python scripts/check_experiment_configs.py <exp>` preflights retention and disk; then `scripts/run_candidates_v2.sh <exp> <min_tok_s> "<keep names>"` runs `configs/<exp>/candidates.yaml` one candidate at a time through `tnlm sweep`, and stops at the first failure.
Each candidate gets a frozen source/config snapshot and a throughput canary, and model states are pruned after evaluation.
Compare results with `tnlm compare_evaluations --pairs <pairs.yaml>` and `scripts/print_pairs.py`.

## Conventions
- One authoritative implementation per operation. Fast paths (`attention_backend: triton`, `loss_chunk_tokens`,
  `activation_checkpointing`) must have tests showing they match the reference.
- Numerics: BF16 compute, at least FP32 for reductions and optimizer state. Helpers upcast but never downcast
  (`normalization.upcast`), so FP64 reference tests stay exact.
- Promote a scratch script into `tnlm/` once it's reused; give it tests when you do.
