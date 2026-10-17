"""Analysis probes carry replayable input lineage and metadata-only leakage gates."""
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
from tnlm.cli import _run_analysis
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import PackedSampler, batch_hash, manifest_hash, prepare_data
from tnlm.model import TensorLM
from tnlm.runtime import full_config


def _fixture(tmp_path, *, train_text='training-only-text', validation_texts=('ab', 'cde'),
             training_test_text=None):
    torch.set_num_threads(1)
    training_rows = [{'id': 'train-only', 'text': train_text}]
    fractions = {'train': 1., 'val': 0., 'test': 0., 'calibration': 0.}
    if training_test_text is not None:
        chosen = {}
        for index in range(100):
            name = f'source-{index}'
            split = 'train' if int(hashlib.sha256(name.encode()).hexdigest(), 16) < 2**255 else 'test'
            chosen.setdefault(split, name)
        training_rows = [{'id': chosen['train'], 'text': train_text},
                         {'id': chosen['test'], 'text': training_test_text}]
        fractions = {'train': .5, 'val': 0., 'test': .5, 'calibration': 0.}
    source = tmp_path / 'train.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in training_rows))
    training = prepare_data(tmp_path / 'training', source_jsonl=source, revision='analysis-training',
                            tokenizer=FixtureTokenizer(), split_fractions=fractions)
    validation_source = tmp_path / 'validation.jsonl'
    validation_source.write_text(''.join(json.dumps({'id': f'val-{i}', 'text': text}) + '\n'
                                         for i, text in enumerate(validation_texts)))
    validation = prepare_data(tmp_path / 'validation', source_jsonl=validation_source,
        revision='analysis-validation', tokenizer=FixtureTokenizer(),
        split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                  d_mlp=12, padded_vocab=50257, attention_backend='reference'),
                 train=TrainConfig(compile=False, bf16_compute=False),
                 raw={'data': {'manifest': str(training), 'context': 2, 'micro_batch_size': 2},
                      'evaluation': {'validation_manifest': str(validation), 'seed': 7},
                      'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0.,
                                  'spend_ledger': str(tmp_path / 'ledger.json')}})
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, TensorLM(cfg.model), None, None, config=full_config(cfg),
                    progress={'steps': 0, 'tokens': 0}, manifest_hash=manifest_hash(training))
    args = argparse.Namespace(checkpoint=str(checkpoint), out=str(tmp_path / 'analysis'),
                              context=None, batches=3, max_seconds=60, device='cpu', run_budget_usd=None)
    return cfg, args, training, validation


def _diagnostic(monkeypatch, callback):
    module = types.ModuleType('tnlm.analysis')
    module.analyze = callback
    monkeypatch.setitem(sys.modules, 'tnlm.analysis', module)


def test_disjoint_probe_plan_replays_batchsize_and_epoch_and_overwrites_opaque_provenance(tmp_path, monkeypatch):
    cfg, args, training, validation = _fixture(tmp_path, training_test_text='ZZZZ-final-test-only')
    before = copy.deepcopy(cfg.to_dict())
    sampler = PackedSampler(validation, split='val', batch_size=2, context=2, seed=7)
    initial = sampler.state_dict()
    expected_batches, mappings, hashes = [], [], []
    for _ in range(3):
        batch = sampler.next_batch()
        expected_batches.append(batch)
        mappings.append(sampler.batch_mappings())
        hashes.append(batch_hash(batch))
    called = []
    def analyze(model, batches, out):
        called.append(True)
        assert len(batches) == 3
        for actual, expected in zip(batches, expected_batches):
            assert actual[0].shape == (2, 2)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            assert not (actual[0] == ord('Z')).any()  # Final-test-only tokens never supplied.
        return {'evaluation_provenance': {'probe_plan_sha256': 'forged',
                'data_overlap_audit': {'has_overlap': True}}, 'diagnostic': 'opaque fixture'}
    _diagnostic(monkeypatch, analyze)
    result = _run_analysis(cfg, args)
    assert called == [True]
    provenance = result['evaluation_provenance']
    plan = provenance['probe_plan']
    assert plan == {'version': 1, 'validation_manifest_sha256': manifest_hash(validation),
                    'split': 'val', 'context': 2, 'sampler_initial': initial,
                    'batch_mappings': mappings, 'batch_hashes': hashes}
    expected_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert provenance['probe_plan_sha256'] == expected_hash
    assert provenance['batch_hashes'] == hashes
    assert {row['epoch'] for batch in mappings for row in batch} == {0, 1}
    selected = {item['document_id'] for batch in mappings for row in batch for item in row['ranges']
                if item['packed_token_start'] < row['input_length']}
    assert provenance['data_overlap_audit']['selected_validation_documents'] == len(selected) == 2
    assert not provenance['data_overlap_audit']['has_overlap']
    assert provenance['data_overlap_audit']['training_manifest_hash'] == manifest_hash(training)
    assert provenance['data_overlap_audit']['validation_manifest_hash'] == manifest_hash(validation)
    assert result['billing']['status'] == 'complete'
    saved = json.loads((tmp_path / 'analysis' / 'invocation.json').read_text())
    assert saved['probe_plan'] == plan and saved['probe_plan_sha256'] == expected_hash
    assert saved['data_overlap_audit'] == provenance['data_overlap_audit']
    assert cfg.to_dict() == before


@pytest.mark.parametrize('kind', ['train-exact', 'train-processed', 'selected-val-duplicates', 'final-test'])
def test_probe_leakage_rejects_before_opaque_callback_with_failed_billing(tmp_path, monkeypatch, kind):
    options = {}
    if kind == 'train-exact':
        options = {'train_text': 'same-text', 'validation_texts': ('same-text', 'unique')}
    elif kind == 'train-processed':
        options = {'train_text': 'line\r\nnext', 'validation_texts': ('line\nnext', 'unique')}
    elif kind == 'selected-val-duplicates':
        options = {'validation_texts': ('duplicate-text', 'duplicate-text')}
    elif kind == 'final-test':
        options = {'training_test_text': 'held-out-secret', 'validation_texts': ('held-out-secret', 'unique')}
    cfg, args, _, _ = _fixture(tmp_path, **options)
    args.context = 64  # Both source documents are selected as actual inputs.
    called = []
    def forbidden(*args):
        called.append(True)
        raise AssertionError('opaque diagnostics must not receive leaking probes')
    _diagnostic(monkeypatch, forbidden)
    with pytest.raises(ValueError, match='overlap|duplicate|leak'):
        _run_analysis(cfg, args)
    assert not called
    assert not (Path(args.out) / 'analysis-invocation-result.json').exists()
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    entry = ledger['runs'][str(Path(args.out).resolve())]
    assert entry['status'] == 'failed' and entry['reserved_usd'] == 0


def test_lookahead_only_document_is_not_analyzed_input_selection(tmp_path, monkeypatch):
    cfg, args, training, validation = _fixture(tmp_path,
        validation_texts=('x', 'training-only-text'))
    cfg.raw['data']['micro_batch_size'] = 1
    for seed in range(20):
        sampler = PackedSampler(validation, split='val', batch_size=1, context=2, seed=seed)
        if sampler.docs[sampler.order[0]]['id'] == 'val-0':
            break
    else:
        raise AssertionError('fixture order unavailable')
    cfg.raw['evaluation']['seed'] = seed
    checkpoint = tmp_path / 'lookahead-checkpoint'
    save_checkpoint(checkpoint, TensorLM(cfg.model), None, None, config=full_config(cfg),
                    progress={'steps': 0, 'tokens': 0}, manifest_hash=manifest_hash(training))
    args.checkpoint, args.batches = str(checkpoint), 1
    called = []
    def analyze(model, batches, out):
        called.append(True)
        assert batches[0][0].tolist() == [[ord('x'), 50256]]
        return {'diagnostic': 'lookahead selection'}
    _diagnostic(monkeypatch, analyze)
    result = _run_analysis(cfg, args)
    provenance = result['evaluation_provenance']
    row = provenance['probe_plan']['batch_mappings'][0][0]
    assert row['ranges'][1]['document_id'] == 'val-1'
    assert row['ranges'][1]['packed_token_start'] == row['input_length'] == 2
    assert provenance['data_overlap_audit']['selected_validation_documents'] == 1
    assert not provenance['data_overlap_audit']['has_overlap']
    assert called == [True]


@pytest.mark.parametrize('tensor_index', [0, 1, 2])
def test_opaque_analyzer_cannot_mutate_probe_tensors(tmp_path, monkeypatch, tensor_index):
    cfg, args, _, _ = _fixture(tmp_path)
    called = []
    def mutate(model, batches, out):
        called.append(True)
        batches[0][tensor_index][0, 0] += 1
        return {'diagnostic': 'mutated caller-owned batch'}
    _diagnostic(monkeypatch, mutate)
    with pytest.raises((RuntimeError, ValueError), match='batch|probe|input|mutat'):
        _run_analysis(cfg, args)
    assert called == [True]
    assert not (Path(args.out) / 'analysis-invocation-result.json').exists()
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    assert ledger['runs'][str(Path(args.out).resolve())]['status'] == 'failed'
