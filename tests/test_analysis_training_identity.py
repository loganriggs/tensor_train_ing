"""Analysis invocation identity must describe its checkpoint training, not its probe."""
import argparse
import copy
import hashlib
import json
import sys
import types

import pytest
import torch

from helpers import FixtureTokenizer
from tnlm.checkpoint import save_checkpoint
from tnlm.cli import _run_analysis
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import manifest_hash, prepare_data
from tnlm.evaluate import _training_descriptor
from tnlm.model import TensorLM
from tnlm.runtime import full_config


@pytest.mark.parametrize('legacy_defaults', [False, True])
@pytest.mark.parametrize('scaled', [False, True])
@pytest.mark.parametrize('context_override', [None, 2])
def test_analysis_uses_checkpoint_training_identity_and_preserves_provenance(
        tmp_path, monkeypatch, scaled, context_override, legacy_defaults):
    torch.set_num_threads(1)
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'id': 'train-only', 'text': 'abcdefghijklm'}) + '\n')
    training = prepare_data(tmp_path / 'training', source_jsonl=source, revision='offline-training',
        tokenizer=FixtureTokenizer(), split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    validation_source = tmp_path / 'validation.jsonl'
    validation_source.write_text(json.dumps({'id': 'validation-only', 'text': 'nopqrstuvwxyz'}) + '\n')
    validation = prepare_data(tmp_path / 'validation', source_jsonl=validation_source, revision='offline-validation',
        tokenizer=FixtureTokenizer(), split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    model_config = ModelConfig(layers=2 if scaled else 1, d_model=16 if scaled else 8,
        heads=1, d_key=4, d_value=4, d_mlp=12, padded_vocab=50257,
        attention_backend='reference', width_scaling_base=8 if scaled else None,
        width_lr_exponent=.25, width_logit_exponent=.75,
        depth_scaling_base=1 if scaled else None, depth_lr_exponent=.3,
        depth_residual_base=1 if scaled else None, depth_residual_exponent=.8, residual_alpha=.6)
    cfg = Config(model=model_config,
        train=TrainConfig(optimizer='adamw' if scaled else 'muon', rung_total_tokens=100,
                          global_prediction_tokens_per_update=7, compile=False, bf16_compute=False),
        raw={'data': {'manifest': str(training), 'context': 4, 'micro_batch_size': 1},
             'evaluation': {'validation_manifest': str(validation)},
             'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0., 'spend_ledger': str(tmp_path / 'ledger.json')}})
    model = TensorLM(model_config)
    progress = {'steps': 11, 'tokens': 77, 'scheduler_step': 11, 'warmup_steps': 137 if scaled else 53}
    extra = {'cooldown_steps': None}
    config = full_config(cfg)
    defaulted = {}
    if legacy_defaults:
        # These added fields are default-valued even in the scaled fixture.
        for name in ('loss_chunk_tokens', 'activation_checkpointing'):
            defaulted['model.' + name] = config['model'].pop(name)
    expected = _training_descriptor({'config': config, 'progress': progress, 'extra': extra,
        'config_compatibility': {'rule': 'missing dataclass field requires current default',
                                 'defaulted_fields': defaulted}})
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, model, None, None, config=config, progress=progress,
                    manifest_hash=manifest_hash(training), extra=extra)
    state_hash = hashlib.sha256((checkpoint / 'state.pt').read_bytes()).hexdigest()
    cfg_before = copy.deepcopy(cfg.to_dict())
    module = types.ModuleType('tnlm.analysis')
    def diagnostic(loaded, batches, out):
        assert loaded.cfg.d_model == model_config.d_model
        assert all(batch[0].shape[1] == (context_override or 4) for batch in batches)
        # Checkpoint-derived wrapper evidence must replace opaque diagnostic metadata.
        return {'diagnostic': 'identity fixture', 'training_descriptor': {'untrusted': True}}
    module.analyze = diagnostic
    monkeypatch.setitem(sys.modules, 'tnlm.analysis', module)
    out = tmp_path / 'analysis'
    args = argparse.Namespace(context=context_override, batches=1, max_seconds=60,
                              checkpoint=str(checkpoint), out=str(out), device='cpu', run_budget_usd=None)
    result = _run_analysis(cfg, args)
    descriptor = result['training_descriptor']
    assert descriptor == expected
    assert descriptor['width'] == (16 if scaled else 8)
    assert descriptor['depth'] == (2 if scaled else 1)
    assert descriptor['optimizer'] == ('adamw' if scaled else 'muon')
    assert descriptor['warmup_steps'] == progress['warmup_steps']
    assert descriptor['width_scaling_base'] == (8 if scaled else 'disabled')
    assert descriptor['depth_scaling_base'] == (1 if scaled else 'disabled')
    assert descriptor['depth_residual_base'] == (1 if scaled else 'disabled')
    assert descriptor['depth_residual_exponent'] == .8 and descriptor['residual_alpha'] == .6
    assert descriptor['train_context'] == 4
    provenance = result['evaluation_provenance']
    assert provenance['context'] == (context_override or 4)
    assert provenance['training_descriptor'] == descriptor
    assert json.loads((out / 'invocation.json').read_text())['training_descriptor'] == descriptor
    assert json.loads((out / 'analysis-invocation-result.json').read_text())['training_descriptor'] == descriptor
    assert provenance['frozen_state']['unchanged'] and result['billing']['status'] == 'complete'
    assert cfg.to_dict() == cfg_before
    assert hashlib.sha256((checkpoint / 'state.pt').read_bytes()).hexdigest() == state_hash
