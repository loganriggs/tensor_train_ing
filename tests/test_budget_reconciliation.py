import json

import pytest

from tnlm.budget import SpendLedger


def setup_ledger(tmp_path):
    ledger = SpendLedger(tmp_path / 'ledger.json', total_budget=100, reserve=0)
    ledger.begin('/identified/run', 36, 100)
    return ledger


def reconcile(ledger, **kwargs):
    inspected = ledger.inspect_reconciliation('/identified/run')
    return ledger.reconcile_terminal('/identified/run', expected_row_sha256=inspected['row_sha256'],
        operator='operator', reason='confirmed stopped on instance',
        terminal_attestation={'run_id': '/identified/run', 'all_workers_stopped': True}, **kwargs)


def test_inspection_lower_bound_and_terminal_total(tmp_path):
    ledger = setup_ledger(tmp_path)
    evidence = tmp_path / 'metrics.jsonl'
    evidence.write_text(json.dumps({'run_id': '/identified/run', 'run_elapsed_seconds': 20}) + '\n')
    original = ledger.path.read_bytes()
    inspected = ledger.inspect_reconciliation('/identified/run', evidence_paths=[evidence])
    assert ledger.path.read_bytes() == original
    assert inspected['observed_cost_lower_bound_usd'] == .2
    assert inspected['evidence'][0]['identity_verified']
    partial = reconcile(ledger, evidence_paths=[evidence])
    assert partial['row']['status'] == 'reconciled_unknown_tail'
    assert partial['row']['spent_usd'] + partial['row']['reserved_usd'] == 1
    final = reconcile(ledger, total_elapsed_seconds=120)
    assert final['row']['spent_usd'] == 1.2
    assert final['row']['reserved_usd'] == 0
    assert not final['training_completion_verified']
    assert final['row']['reconciliations'][-1]['old_row_sha256'] == partial['row_sha256']
    with pytest.raises(ValueError, match='unresolved'):
        reconcile(ledger, total_cost_usd=1.2)


def test_stale_digest_and_unattested_rejected_without_changes(tmp_path):
    ledger = setup_ledger(tmp_path)
    inspected = ledger.inspect_reconciliation('/identified/run')
    reconcile(ledger, observed_elapsed_seconds=10)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match='changed'):
        ledger.reconcile_terminal('/identified/run', expected_row_sha256=inspected['row_sha256'],
            operator='a', reason='b', terminal_attestation={'run_id': '/identified/run', 'all_workers_stopped': True})
    with pytest.raises(ValueError, match='attestation'):
        ledger.reconcile_terminal('/identified/run', expected_row_sha256=inspected['row_sha256'],
            operator='a', reason='b', terminal_attestation={'run_id': '/identified/run', 'all_workers_stopped': False})
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize('amount', [float('nan'), float('inf'), -1, True])
def test_invalid_amount_no_mutation(tmp_path, amount):
    ledger = setup_ledger(tmp_path)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        reconcile(ledger, total_cost_usd=amount)
    assert ledger.path.read_bytes() == before


def test_evidence_identity_and_inconsistent_totals(tmp_path):
    ledger = setup_ledger(tmp_path)
    evidence = tmp_path / 'other.json'
    evidence.write_text(json.dumps({'run_id': '/other', 'run_elapsed_seconds': 90}))
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match='identity'):
        reconcile(ledger, evidence_paths=[evidence])
    with pytest.raises(ValueError, match='below'):
        reconcile(ledger, observed_elapsed_seconds=50, total_elapsed_seconds=40)
    with pytest.raises(ValueError, match='one authoritative'):
        reconcile(ledger, total_elapsed_seconds=10, total_cost_usd=.1)
    assert ledger.path.read_bytes() == before


def test_unknown_tail_does_not_free_admission_capacity(tmp_path):
    ledger = setup_ledger(tmp_path)
    reconcile(ledger, observed_elapsed_seconds=20)
    ledger.total = 1
    with pytest.raises(ValueError, match='budget exhausted'):
        ledger.begin('other', 36, 1)
    reconcile(ledger, total_cost_usd=.2)
    ledger.begin('other', 36, 1)
