"""Launch-file guards for configs/h100/*.yaml: a broken invariant should fail here, not on the rented node."""
from pathlib import Path

import pytest

from tnlm.checkpoint_plan import pending_milestone_tokens
from tnlm.config import Config

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "configs/h100").glob("*.yaml"))


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_h100_config_invariants(path):
    c = Config.from_yaml(str(path))
    assert c.model.normalization in ("pre_rms", "spectral_rms")
    if c.model.normalization == "spectral_rms":
        assert c.train.qk_clip_tau == 0.0  # QK-clip rejects spectral models at trainer start
    else:
        assert c.train.qk_clip_tau > 0 and c.train.qk_clip_every_steps > 0
    if c.train.optimizer == "muon":  # make_optimizer rejects Muon + width_scaling_base; Muon's spectral rule does the shape scaling
        assert c.model.width_scaling_base is None and c.train.muon_adjust_lr == "spectral_unclamped"
    else:
        assert c.train.optimizer == "adamw" and c.model.width_scaling_base == 512
    assert c.train.global_prediction_tokens_per_update >= 131072
    assert c.train.compile is False  # the distributed trainer rejects compile (red-team D1, 2026-10-20)
    # No discarded micro-batch rounds: a 4x2048 (or 16x512) micro-batch always holds >= 8160 targets (min 8165-8167 measured), so the quota must be a
    # multiple of 8 ranks x 8160; 262144 read 1.25x the trained tokens (red-team 22:40 Oct 19).
    assert c.raw["data"]["micro_batch_size"] * c.raw["data"]["context"] == 8192
    assert c.train.global_prediction_tokens_per_update % (8 * 8160) == 0
    assert isinstance(c.train.rung_total_tokens, int) and c.train.rung_total_tokens > 1e10
    kept = sum(1 for _ in pending_milestone_tokens(c, start_tokens=0, token_limit=c.train.rung_total_tokens))
    assert kept <= 12, "sparse milestones required (the default keeps a full state every 100M tokens)"
    assert c.raw["project"].get("keep_recovery_checkpoints") == 2


def test_h100_variants_differ_only_in_normalization_and_clip():
    a, b = (Config.from_yaml(str(ROOT / f"configs/h100/{n}.yaml")).raw for n in ("1b_pre_rms", "1b_spectral"))
    for sec in ("model", "train", "data", "project"):
        diff = {k for k in set(a[sec]) | set(b[sec]) if a[sec].get(k) != b[sec].get(k)}
        assert diff <= {"normalization", "qk_clip_tau", "qk_clip_every_steps"}, (sec, diff)


def test_h100_muon_variant_differs_from_b_only_in_optimizer_settings():
    b, c = (Config.from_yaml(str(ROOT / f"configs/h100/{n}.yaml")).raw for n in ("1b_spectral", "1b_spectral_muon"))
    allowed = {"model": {"width_scaling_base"}, "train": {"optimizer", "muon_adjust_lr", "aux_adamw_lr", "lr"}, "data": set(), "project": set()}
    for sec, ok in allowed.items():
        diff = {k for k in set(b[sec]) | set(c[sec]) if b[sec].get(k) != c[sec].get(k)}
        assert diff <= ok, (sec, diff - ok)
