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
    assert c.model.width_scaling_base == 512 and c.train.global_prediction_tokens_per_update >= 131072
    assert isinstance(c.train.rung_total_tokens, int) and c.train.rung_total_tokens > 1e10
    kept = sum(1 for _ in pending_milestone_tokens(c, start_tokens=0, token_limit=c.train.rung_total_tokens))
    assert kept <= 12, "sparse milestones required (the default keeps a full state every 100M tokens)"
    assert c.raw["project"].get("keep_recovery_checkpoints") == 2


def test_h100_variants_differ_only_in_normalization_and_clip():
    a, b = (Config.from_yaml(str(ROOT / f"configs/h100/{n}.yaml")).raw for n in ("1b_pre_rms", "1b_spectral"))
    for sec in ("model", "train", "data", "project"):
        diff = {k for k in set(a[sec]) | set(b[sec]) if a[sec].get(k) != b[sec].get(k)}
        assert diff <= {"normalization", "qk_clip_tau", "qk_clip_every_steps"}, (sec, diff)
