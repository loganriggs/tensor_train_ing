import json
import pytest
from tnlm.budget import SpendLedger
from tnlm.cli import main


def test_inspect_then_attested_lower_bound_then_total(tmp_path, capsys):
    path = tmp_path/'ledger.json';run = str(tmp_path/'run')
    ledger = SpendLedger(path);ledger.begin(run, 1., 3600)
    metrics = tmp_path/'metrics.jsonl'
    metrics.write_text(json.dumps({'run_id':run,'run_elapsed_seconds':1800})+'\n')
    args = ['reconcile','--ledger',str(path),'--run-id',run]
    before = path.read_bytes()
    assert main(args+['--metrics',str(metrics)]) == 0
    view = json.loads(capsys.readouterr().out)
    assert path.read_bytes() == before and not view['termination_verified']
    assert view['observed_cost_lower_bound_usd'] == .5
    apply = ['--apply','--operator','test','--reason','identified worker terminated','--all-workers-stopped']
    assert main(args+apply+['--expected-row-sha256',view['row_sha256'],'--metrics',str(metrics)]) == 0
    lower = json.loads(capsys.readouterr().out)
    assert lower['row']['status'] == 'reconciled_unknown_tail'
    assert lower['row']['spent_usd'] == lower['row']['reserved_usd'] == .5
    assert main(args+apply+['--expected-row-sha256',lower['row_sha256'],'--total-elapsed-seconds','2700']) == 0
    final = json.loads(capsys.readouterr().out)
    assert final['row']['spent_usd'] == .75 and final['row']['reserved_usd'] == 0
    assert not final['training_completion_verified']


@pytest.mark.parametrize('extra',[['--apply'],['--total-elapsed-seconds','2'],['--all-workers-stopped']])
def test_incomplete_or_ignored_attestation_rejected_before_write(tmp_path, extra):
    path=tmp_path/'ledger.json';ledger=SpendLedger(path);ledger.begin('run',1,10)
    before=path.read_bytes()
    with pytest.raises(SystemExit): main(['reconcile','--ledger',str(path),'--run-id','run']+extra)
    assert path.read_bytes()==before
