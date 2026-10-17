"""Real final-position activation transfers on immutable exact-label task probes."""
import json

import pytest
import torch
import torch.nn.functional as F

from tnlm.checkpoint import load_checkpoint
from tnlm.evaluate import evaluate_mechanisms
from tnlm.model import TensorLM
from tnlm.tasks import load_mechanism_tasks


def test_reuse_matches_independent_manual_patch_across_prompt_lengths(checkpoint_fixture, tmp_path):
    import tiktoken
    cfg, checkpoint, tasks = checkpoint_fixture
    result = evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'reuse', max_examples=16)
    _, rows = load_mechanism_tasks(tasks)
    by_id = {row['id']: row for row in rows}
    tokenizer = tiktoken.get_encoding('gpt2')
    prompts = {row['id']: torch.tensor([tokenizer.encode_ordinary(row['prompt'])]) for row in rows}
    model = TensorLM(cfg.model).eval()
    load_checkpoint(checkpoint, model, restore_rng=False)
    cached = {}
    with torch.inference_mode():
        for row in rows:
            captures = {}
            handles = []
            for branch in ('attn', 'mlp'):
                def capture(module, args, output, branch=branch):
                    captures[branch] = output[0, -1].clone()
                handles.append(getattr(model.blocks[0], branch).register_forward_hook(capture))
            try:
                model(prompts[row['id']], doc_ids=torch.zeros_like(prompts[row['id']]))
            finally:
                for handle in handles:
                    handle.remove()
            cached[row['id']] = captures
        saw_different_lengths = False
        patched_count = 0
        for target in result['examples']:
            for component, transfers in target['branch_reuse'].items():
                branch = component.split(':')[1]
                for donor_class, measured in transfers.items():
                    assert measured['status'] == 'patched'
                    donor = by_id[measured['donor_id']]
                    assert donor['task'] == target['task']
                    assert donor['composition_id'] != target['composition_id']
                    assert donor['id'] != target['id']
                    assert (donor['answer'] == target['answer']) == (donor_class == 'same_answer')
                    eligible = [row for row in rows if row['task'] == target['task']
                                and row['composition_id'] != target['composition_id']
                                and ((row['answer'] == target['answer']) == (donor_class == 'same_answer'))]
                    assert donor['id'] == min(row['id'] for row in eligible)
                    if prompts[donor['id']].shape[1] != prompts[target['id']].shape[1]:
                        saw_different_lengths = True
                    vector = cached[donor['id']][branch]
                    def patch(module, args, output):
                        patched = output.clone()
                        patched[0, -1] = vector
                        assert torch.equal(patched[:, :-1], output[:, :-1])
                        return patched
                    handle = getattr(model.blocks[0], branch).register_forward_hook(patch)
                    try:
                        idx = prompts[target['id']]
                        logits = model(idx, doc_ids=torch.zeros_like(idx))[0, -1].float()
                    finally:
                        handle.remove()
                    ce = float(F.cross_entropy(logits[None], torch.tensor([target['answer_token']])))
                    candidates = result['probe']['candidate_token_ids'][target['task']]
                    assert measured['answer_ce'] == pytest.approx(ce, abs=1e-7)
                    assert measured['full_vocab_correct'] == (int(logits.argmax()) == target['answer_token'])
                    assert measured['candidate_correct'] == (candidates[int(logits[candidates].argmax())] == target['answer_token'])
                    assert measured['delta_answer_ce'] == pytest.approx(ce - target['baseline']['answer_ce'], abs=1e-7)
                    patched_count += 1
        assert saw_different_lengths
    assert result['exposure']['reuse_answer_prediction_tokens'] == patched_count == 64
    assert result['exposure']['total_answer_prediction_tokens'] == 112
    assert result['probe']['reuse_policy']['cached_vector_elements'] == 16 * 2 * cfg.model.d_model
    assert result['frozen_state']['unchanged']
    assert result['summary']['branch_reuse']['0:mlp']['same_answer']['patched_examples'] == 16
    assert result['summary']['branch_reuse']['0:mlp']['different_answer']['patched_examples'] == 16
    assert result['billing']['status'] == 'complete'


def test_donor_selection_reproducible_and_missing_donors_explicit(checkpoint_fixture, tmp_path):
    cfg, checkpoint, tasks = checkpoint_fixture
    a = evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'first', max_examples=8, components=('0:mlp',))
    b = evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'second', max_examples=8, components=('0:mlp',))
    assert [row['branch_reuse'] for row in a['examples']] == [row['branch_reuse'] for row in b['examples']]
    single = evaluate_mechanisms(cfg, checkpoint, tasks, tmp_path / 'single', max_examples=1)
    assert single['exposure']['reuse_answer_prediction_tokens'] == 0
    for transfers in single['examples'][0]['branch_reuse'].values():
        for measured in transfers.values():
            assert measured['status'] == 'skipped'
            assert 'no same-task different-composition donor' in measured['reason']
    for transfers in single['summary']['branch_reuse'].values():
        for summary in transfers.values():
            assert summary['patched_examples'] == 0 and summary['skipped_examples'] == 1


def test_patch_failure_removes_hooks_restores_flags_and_bills(checkpoint_fixture, tmp_path, monkeypatch):
    cfg, checkpoint, tasks = checkpoint_fixture
    model = TensorLM(cfg.model)
    model.blocks[0].ablate = frozenset({'attn'})
    original_forward = model.forward

    def fail_during_patch(*args, **kwargs):
        if any(hook.__name__ == 'patch' for block in model.blocks
               for module in (block.attn, block.mlp) for hook in module._forward_hooks.values()):
            raise RuntimeError('injected activation-patch failure')
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, 'forward', fail_during_patch)
    monkeypatch.setattr('tnlm.evaluate.TensorLM', lambda cfg: model)
    out = tmp_path / 'failed-patch'
    with pytest.raises(RuntimeError, match='activation-patch failure'):
        evaluate_mechanisms(cfg, checkpoint, tasks, out, max_examples=8, components=('0:mlp',))
    assert model.blocks[0].ablate == frozenset({'attn'})
    assert not model.blocks[0].attn._forward_hooks
    assert not model.blocks[0].mlp._forward_hooks
    assert not (out / 'mechanism_evaluation.json').exists()
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    row = ledger['runs'][str(out.resolve())]
    assert row['status'] == 'failed' and row['reserved_usd'] == 0
    assert row['spent_usd'] > 0
    json.dumps(row, allow_nan=False)
