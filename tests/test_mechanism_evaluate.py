"""Real tiny CPU checkpoints scored against exact-label GPT2 mechanism probes."""
import json

import pytest
import torch
import torch.nn.functional as F

from tnlm.checkpoint import load_checkpoint, save_checkpoint
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data, prepare_mechanism_tasks, manifest_hash, load_mechanism_tasks
from tnlm.model import TensorLM
from tnlm.train import evaluate_mechanisms, full_config
from helpers import FixtureTokenizer


def test_actual_answer_ce_accuracy_and_branch_contrasts(checkpoint_fixture, tmp_path):
    import tiktoken
    cfg, checkpoint, tasks = checkpoint_fixture
    result = evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'evaluation', max_examples=16)
    assert result['kind'] == 'known_mechanism_evaluation'
    assert set(result['per_task']) == {'boolean_hierarchy', 'associative_retrieval'}
    assert result['summary']['examples'] == 16
    assert result['frozen_state']['unchanged']
    assert result['generator_composition_holdout_verified']
    assert result['benchmark_training_exposure_verified'] is False
    _, rows = load_mechanism_tasks(tasks)
    assert result['probe_ids'] == [row['id'] for row in rows]
    tokenizer = tiktoken.get_encoding('gpt2')
    model = TensorLM(cfg.model).eval()
    load_checkpoint(checkpoint, model, restore_rng=False)
    losses = []
    with torch.inference_mode():
        for expected, measured in zip(rows, result['examples']):
            idx = torch.tensor([tokenizer.encode_ordinary(expected['prompt'])])
            answer = tokenizer.encode_ordinary(expected['answer'])[0]
            digits = range(2) if expected['task'] == 'boolean_hierarchy' else range(4)
            candidates = [tokenizer.encode_ordinary(f' {digit}')[0] for digit in digits]
            for component in [None, '0:attn', '0:mlp']:
                model.blocks[0].ablate = frozenset() if component is None else frozenset({component.split(':')[1]})
                logits = model(idx, doc_ids=torch.zeros_like(idx))[0, -1].float()
                loss = float(F.cross_entropy(logits[None], torch.tensor([answer])))
                report = measured['baseline'] if component is None else measured['branch_removal'][component]
                assert report['answer_ce'] == pytest.approx(loss, abs=1e-7)
                assert report['full_vocab_correct'] == (int(logits.argmax()) == answer)
                assert report['candidate_correct'] == (candidates[int(logits[candidates].argmax())] == answer)
                if component is None:
                    losses.append(loss)
                else:
                    assert report['delta_answer_ce'] == pytest.approx(loss - measured['baseline']['answer_ce'])
    assert result['summary']['baseline']['answer_ce'] == pytest.approx(sum(losses) / len(losses))
    saved = json.loads((tmp_path / 'evaluation' / 'mechanism_evaluation.json').read_text())
    assert saved['billing']['status'] == 'complete'
    assert saved['billing']['spent_usd'] > 0 and saved['billing']['reserved_usd'] == 0
    assert saved['billing']['parent_checkpoint'] == str(checkpoint.resolve())
    assert saved['exposure']['answer_prediction_tokens'] == 48
    assert 'tnlm/model.py' in saved['runtime_source_sha256']


def test_prompt_cap_and_dryrun_do_not_launch(checkpoint_fixture, tmp_path):
    cfg, checkpoint, tasks = checkpoint_fixture
    with pytest.raises(ValueError, match='no truncation'):
        evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'short', max_prompt_tokens=1)
    row = json.loads((tmp_path / 'spend.json').read_text())['runs'][str((tmp_path / 'short').resolve())]
    assert row['status'] == 'failed' and row['reserved_usd'] == 0
    cfg.raw['project']['dry_run'] = True
    previous = (tmp_path / 'spend.json').read_bytes()
    result = evaluate_mechanisms(cfg, 'missing', 'missing', tmp_path / 'dry')
    assert result['dry_run'] and not (tmp_path / 'dry').exists()
    assert previous == (tmp_path / 'spend.json').read_bytes()


def test_injected_branch_failure_restores_flags_and_finalizes_bill(checkpoint_fixture, tmp_path, monkeypatch):
    cfg, checkpoint, tasks = checkpoint_fixture
    model = TensorLM(cfg.model)
    model.blocks[0].ablate = frozenset({'mlp'})
    original = model.forward

    def fail_only_removed_attention(*args, **kwargs):
        if 'attn' in model.blocks[0].ablate:
            raise RuntimeError('injected branch failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(model, 'forward', fail_only_removed_attention)
    monkeypatch.setattr('tnlm.evaluate.TensorLM', lambda cfg: model)
    with pytest.raises(RuntimeError, match='injected branch failure'):
        evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'failed', max_examples=1)
    assert model.blocks[0].ablate == frozenset({'mlp'})
    row = json.loads((tmp_path / 'spend.json').read_text())['runs'][str((tmp_path / 'failed').resolve())]
    assert row['status'] == 'failed' and row['reserved_usd'] == 0
    assert not (tmp_path / 'failed' / 'mechanism_evaluation.json').exists()


def test_unknown_component_rejected_before_admission(checkpoint_fixture, tmp_path):
    cfg, checkpoint, tasks = checkpoint_fixture
    with pytest.raises(ValueError, match='component'):
        evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'bad-component', components=['9:attn'])
    assert not (tmp_path / 'spend.json').exists()
    assert not (tmp_path / 'bad-component').exists()


def test_nonfinite_logits_fail_without_complete_artifact(checkpoint_fixture, tmp_path, monkeypatch):
    cfg, checkpoint, tasks = checkpoint_fixture
    model = TensorLM(cfg.model)
    original = model.forward

    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result[..., 0] = float('nan')
        return result

    monkeypatch.setattr(model, 'forward', nonfinite)
    monkeypatch.setattr('tnlm.evaluate.TensorLM', lambda cfg: model)
    with pytest.raises(FloatingPointError, match='nonfinite mechanism logits'):
        evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'nonfinite', max_examples=1)
    row = json.loads((tmp_path / 'spend.json').read_text())['runs'][str((tmp_path / 'nonfinite').resolve())]
    assert row['status'] == 'failed' and row['reserved_usd'] == 0
    assert model.blocks[0].ablate == frozenset()
    assert not (tmp_path / 'nonfinite' / 'mechanism_evaluation.json').exists()


def test_rejects_checkpoint_with_different_tokenizer(checkpoint_fixture, tmp_path):
    cfg, _, tasks = checkpoint_fixture
    source = tmp_path / 'wrong-tokenizer.jsonl'
    source.write_text(json.dumps({'id': 'wrong-tokenizer', 'text': 'fixture alphabet'}) + '\n')
    training = prepare_data(tmp_path / 'wrong-tokenizer-data', source_jsonl=source,
        revision='wrong-tokenizer-v1', tokenizer=FixtureTokenizer(),
        split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    cfg.raw['data']['manifest'] = str(training)
    checkpoint = tmp_path / 'wrong-tokenizer-checkpoint'
    save_checkpoint(checkpoint, TensorLM(cfg.model), None, None, config=full_config(cfg),
                    progress={'steps': 0}, manifest_hash=manifest_hash(training))
    out = tmp_path / 'wrong-tokenizer-evaluation'
    with pytest.raises(ValueError, match='training GPT2 tokenizer fingerprint'):
        evaluate_mechanisms(cfg, checkpoint, tasks, out, max_examples=1)
    assert not (out / 'mechanism_evaluation.json').exists()
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    assert ledger['runs'][str(out.resolve())]['status'] == 'failed'


@pytest.mark.parametrize('split,verified,status', [('train', True, 'verified'), ('val', False, 'overlap')])
def test_report_binds_declared_training_exposure(checkpoint_fixture, tmp_path, split, verified, status):
    cfg, _, tasks = checkpoint_fixture
    training = prepare_data(tmp_path / 'benchmark-corpus', source_jsonl=tasks.with_name(f'{split}.jsonl'),
                            revision='benchmark-v1',
                            split_fractions={'train': .75, 'val': 0., 'test': 0., 'calibration': .25})
    cfg.raw['data']['manifest'] = str(training)
    checkpoint = tmp_path / 'benchmark-checkpoint'
    save_checkpoint(checkpoint, TensorLM(cfg.model), None, None, config=full_config(cfg),
                    progress={'steps': 0}, manifest_hash=manifest_hash(training))
    result = evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'benchmark-evaluation',
                                 max_examples=2, components=())
    assert result['benchmark_training_exposure_verified'] is verified
    audit = result['training_exposure_audit']
    assert audit['status'] == status
    assert audit['benchmark_manifest_hash'] == result['task_manifest_sha256']
    assert audit['training_manifest_hash'] == result['training_manifest_sha256']
    assert audit['has_validation_overlap'] is (split == 'val')
    assert 'not optimizer visitation' in audit['scope']
