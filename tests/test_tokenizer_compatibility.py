"""Tokenizer identity tests; synthetic fingerprints validate metadata only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from helpers import FixtureTokenizer
from tnlm.checkpoint import save_checkpoint
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import manifest_hash, prepare_data, validate_tokenizer_compatibility
from tnlm.model import TensorLM
from tnlm.runtime import full_config


def synthetic_production():
    # Deliberately synthetic: this metadata is never used to claim real encoding.
    return {'name': 'gpt2', 'eos_id': 50256, 'fingerprint_sha256': 'a' * 64,
            'vocabulary_sha256': 'b' * 64, 'regex_pattern': 'synthetic-test-pattern',
            'special_tokens': {'<|endoftext|>': 50256}, 'package': 'synthetic', 'version': '1'}


def metadata_manifest(tmp_path, name, tokenizer):
    root = tmp_path / name
    root.mkdir()
    path = root / 'manifest.json'
    path.write_text(json.dumps({'tokenizer': tokenizer}))
    checksum(path)
    return path


def checksum(path):
    path.with_name('manifest.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '\n')


def test_synthetic_production_fingerprint_match_allows_package_version_changes(tmp_path):
    left = synthetic_production()
    right = copy.deepcopy(left)
    right.update(package='another-provenance-label', version='2')
    a, b = [metadata_manifest(tmp_path, name, value) for name, value in [('a', left), ('b', right)]]
    result = validate_tokenizer_compatibility(a, b)
    assert result['status'] == 'fingerprint_match'
    assert result['training_manifest_sha256'] == manifest_hash(a)
    assert result['validation_manifest_sha256'] == manifest_hash(b)
    assert result['identity']['fingerprint_sha256'] == 'a' * 64


@pytest.mark.parametrize('key,value', [('name', 'other'), ('eos_id', 0),
    ('fingerprint_sha256', 'c' * 64), ('vocabulary_sha256', 'c' * 64),
    ('regex_pattern', 'different'), ('special_tokens', {'different': 50256})])
def test_synthetic_production_semantic_mismatch_rejected(tmp_path, key, value):
    left = synthetic_production()
    right = {**left, key: value}
    a = metadata_manifest(tmp_path, 'a', left)
    b = metadata_manifest(tmp_path, 'b', right)
    with pytest.raises(ValueError, match='tokenizer'):
        validate_tokenizer_compatibility(a, b)


@pytest.mark.parametrize('defect', ['missing-tokenizer', 'missing-fingerprint', 'bad-fingerprint',
                                   'missing-checksum', 'wrong-checksum'])
def test_missing_or_invalid_identity_and_checksum_reject(tmp_path, defect):
    a = metadata_manifest(tmp_path, 'a', synthetic_production())
    b = metadata_manifest(tmp_path, 'b', synthetic_production())
    value = json.loads(b.read_text())
    if defect == 'missing-tokenizer':
        value.pop('tokenizer')
    elif defect == 'missing-fingerprint':
        value['tokenizer'].pop('fingerprint_sha256')
    elif defect == 'bad-fingerprint':
        value['tokenizer']['fingerprint_sha256'] = 'not-a-sha256'
    b.write_text(json.dumps(value))
    checksum(b)
    if defect == 'missing-checksum':
        b.with_name('manifest.sha256').unlink()
    elif defect == 'wrong-checksum':
        b.with_name('manifest.sha256').write_text('0' * 64)
    with pytest.raises((ValueError, OSError)):
        validate_tokenizer_compatibility(a, b)


def test_explicit_matching_fixture_allowed_only_with_optin_and_never_production_mix(tmp_path):
    fixture = {'name': 'offline', 'eos_id': 50256, 'injected': True, 'test_only': True}
    a = metadata_manifest(tmp_path, 'a', fixture)
    b = metadata_manifest(tmp_path, 'b', fixture)
    with pytest.raises(ValueError, match='CPU'):
        validate_tokenizer_compatibility(a, b)
    result = validate_tokenizer_compatibility(a, b, allow_test_only=True)
    assert result['status'] == 'test_only_metadata_match'
    assert 'not production' in result['scope']
    production = metadata_manifest(tmp_path, 'production', synthetic_production())
    for left, right in [(a, production), (production, a)]:
        with pytest.raises(ValueError):
            validate_tokenizer_compatibility(left, right, allow_test_only=True)


def test_real_gpt2_preparation_identity_smoke(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'id': 'real-tokenizer', 'text': 'A tiny GPT2 tokenization smoke test.'}) + '\n')
    a = prepare_data(tmp_path / 'real-a', source_jsonl=source, revision='offline-real-tokenizer')
    b = prepare_data(tmp_path / 'real-b', source_jsonl=source, revision='offline-real-tokenizer')
    result = validate_tokenizer_compatibility(a, b)
    assert result['status'] == 'fingerprint_match'
    assert result['identity']['name'] == 'gpt2' and result['identity']['eos_id'] == 50256
    assert len(result['identity']['fingerprint_sha256']) == 64


@pytest.mark.parametrize('entrypoint', ['evaluate', 'analysis'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_entrypoint_rejects_incompatible_or_cuda_fixture_before_sampler_model_callback(
        tmp_path, monkeypatch, entrypoint, device):
    torch.set_num_threads(1)
    manifests = {}
    for split, text in [('train', 'abcdefgh'), ('val', 'nopqrstu')]:
        source = tmp_path / f'{split}.jsonl'
        source.write_text(json.dumps({'id': split, 'text': text}) + '\n')
        manifests[split] = prepare_data(tmp_path / split, source_jsonl=source, revision='offline-fixture',
            tokenizer=FixtureTokenizer(), split_fractions={key: float(key == split)
            for key in ('train', 'val', 'test', 'calibration')})
    cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                  d_mlp=12, padded_vocab=50257, attention_backend='reference'),
                 train=TrainConfig(compile=False, bf16_compute=False),
                 raw={'data': {'manifest': str(manifests['train']), 'context': 4},
                      'evaluation': {'validation_manifest': str(manifests['val'])},
                      'project': {'dry_run': False, 'gpu_hourly_rate_usd': .01,
                                  'spend_ledger': str(tmp_path / 'ledger.json')}})
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, TensorLM(cfg.model), None, None, config=full_config(cfg),
                    progress={'steps': 0, 'tokens': 0}, manifest_hash=manifest_hash(manifests['train']))
    if device == 'cpu':
        value = json.loads(manifests['val'].read_text())
        value['tokenizer']['name'] = 'incompatible-offline-fixture'
        manifests['val'].write_text(json.dumps(value))
        checksum(manifests['val'])
    else:
        # Only availability reporting is mocked. Rejection must precede CUDA
        # allocation and every model/sampler/diagnostic call; no GPU work runs.
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    import tnlm.cli as cli
    import tnlm.evaluate as evaluate
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('tokenizer gate must precede sampler/model/diagnostic')
    monkeypatch.setattr(cli, 'TensorLM', forbidden)
    monkeypatch.setattr(evaluate, 'TensorLM', forbidden)
    monkeypatch.setattr('tnlm.data.PackedSampler', forbidden)
    module = types.ModuleType('tnlm.analysis')
    module.analyze = forbidden
    monkeypatch.setitem(sys.modules, 'tnlm.analysis', module)
    out = tmp_path / 'output'
    with pytest.raises(ValueError, match='tokenizer|CPU'):
        if entrypoint == 'evaluate':
            evaluate.evaluate_checkpoint(cfg, checkpoint, out, max_seconds=60, device=device,
                                         max_prediction_tokens=7, bootstrap_samples=0)
        else:
            cli._run_analysis(cfg, argparse.Namespace(checkpoint=str(checkpoint), out=str(out),
                context=4, batches=1, max_seconds=60, device=device, run_budget_usd=None))
    assert calls == []
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    assert ledger['runs'][str(out.resolve())]['status'] == 'failed'
    assert not (out / 'evaluation.json').exists()
    assert not (out / 'analysis-invocation-result.json').exists()
