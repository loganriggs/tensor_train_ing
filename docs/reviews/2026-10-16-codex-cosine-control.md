# Fixed-horizon cosine finalist control

The research brief§5 asks for a full-cosine cross-check when finalists are close. Set `train.schedule: cosine` for this control. The default `constant_with_separate_cooldown` remains unchanged.

Let D be the requested prediction-token endpoint, B the target tokens per update, N=ceil(D/B), and W=max(100,ceil(.01N)). A cosine run requires N>W. For zero-based optimizer update j, the multiplier is:

- `(j+1)/W` for j<W;
- `0.5 * (1 + cos(pi * (j+1-W)/(N-W)))` otherwise.

This is a single cosine decay after warmup, with no restarts. The cosine form follows [SGDR, Loshchilov and Hutter](https://arxiv.org/abs/1608.03983); the warmup rule and discrete endpoint convention above are this project's explicit choices. The final update uses LR0; optimizer moment/statistic accounting still executes for its targets. Each optimizer group retains its own base LR, including Muon auxiliary AdamW and width-scaled groups.

A resumed cosine run must retain its exact original token horizon. Changing D would alter the decay path and is rejected. Constant-LR upward promotion remains available under the previous contract. Separate cooldown forks from cosine are rejected. The default mandatory90% branch save is suppressed for cosine; ordinary configured milestones and recovery/final checkpoints remain.

For a sweep use `--no-cooldown` with cosine candidates. Reporting keeps `schedule: cosine` explicit. `cooldown_status: uncooled` means no separate cooldown fork; it does not mean a cosine run used constant LR. Completion requires consistent token and scheduler-step endpoints. Mixed schedule comparisons remain explicit contrasts without automatic ranking certification.

This implementation enables the experiment; it does not establish that cosine improves a finalist or that short-run rankings transfer. No funded cosine run was launched by this change.

Early resume compatibility preflight now checks serial checkpoint size/hash before deserializing its configuration; canonical restoration still performs the full identity checks. Distributed dry-run planning does not read a placeholder parent path.

Validation:507 CPU tests passed (3warnings,139.50s), including exact AdamW/Muon and two-rank CPU/gloo full/resume replay across warmup into decay, final partial token update, planning/storage/reporting and pre-admission failures. Claude GPU kernel/parallel tests were excluded; CUDA/NCCL behavior is not established by this gate.
