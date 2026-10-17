import json
import pytest
import yaml
from tnlm.sweep import run_sweep
from tnlm.cli import main


def execute(config, out, **kwargs):
    return run_sweep(config, [{'name':'a','overrides':{}}], out,
                     max_seconds=60, evaluation_max_seconds=30, device='cpu',
                     cooldown=False, max_prediction_tokens=14, bootstrap_samples=0, **kwargs)


def test_dry_plan_freezes_canary_and_rejects_short_horizon(config, tmp_path):
    config.raw['project']['dry_run']=True
    out=tmp_path/'out'
    result=execute(config,out,canary_steps=2,min_tokens_per_second=10,canary_warmup_steps=1)
    policy=result['identity']['controls']['throughput_canary']
    assert policy['measured_steps']==2 and policy['warmup_steps']==1
    assert policy['min_tokens_per_second']==10
    with pytest.raises(ValueError,match='horizon'):
        execute(config,out,canary_steps=100,min_tokens_per_second=10)
    assert not out.exists()


@pytest.mark.parametrize('kwargs',[{'canary_steps':True}, {'min_tokens_per_second':10},
    {'canary_steps':1}, {'canary_steps':1,'min_tokens_per_second':float('nan')},
    {'canary_steps':1,'min_tokens_per_second':1,'canary_warmup_steps':-1}])
def test_invalid_controls_before_artifacts(config,tmp_path,kwargs):
    with pytest.raises(ValueError): execute(config,tmp_path/'bad',**kwargs)
    assert not (tmp_path/'bad').exists()


def test_real_failed_canary_billed_no_evaluation_or_retry(config,tmp_path,monkeypatch):
    import tnlm.sweep as sweep
    monkeypatch.setattr(sweep,'evaluate_checkpoint',lambda *a,**k:pytest.fail('evaluation launched'))
    out=tmp_path/'failed'
    args=dict(canary_steps=1,min_tokens_per_second=1e20,canary_warmup_steps=0)
    with pytest.raises(RuntimeError,match='[Cc]anary|throughput'):execute(config,out,**args)
    journal=json.loads((out/'sweep.json').read_text())
    assert journal['stages']['a/parent']['status']=='failed'
    entries=json.loads((tmp_path/'spend.json').read_text())['runs']
    assert len(entries)==1 and next(iter(entries.values()))['status']=='failed'
    assert (out/'a/parent/canary.json').is_file()
    with pytest.raises(RuntimeError,match='automatic retry forbidden'):execute(config,out,**args)
    assert json.loads((tmp_path/'spend.json').read_text())['runs']==entries


def test_snapshot_forwards_canary_controls(config,tmp_path,monkeypatch):
    import tnlm.source_snapshot as snapshot
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config.to_dict()))
    grid=tmp_path/'grid.yaml';grid.write_text('- name: a\n  overrides: {}\n')
    monkeypatch.setattr(snapshot,'create_snapshot',lambda *a,**k:None)
    def run(destination,child,**kwargs):
        assert child[child.index('--canary-steps')+1]=='2'
        assert child[child.index('--canary-warmup-steps')+1]=='1'
        assert child[child.index('--min-tok-s')+1]=='100.0'
        return 0
    monkeypatch.setattr(snapshot,'run_snapshot',run)
    assert main(['sweep','--config',str(path),'--candidates',str(grid),'--out',str(tmp_path/'out'),
                 '--max-seconds','60','--evaluation-max-seconds','30','--device','cpu',
                 '--snapshot-sources','--canary-steps','2','--canary-warmup-steps','1',
                 '--expect-tok-s','100'])==0


def test_real_passing_canary_continues_same_parent_and_evaluates(config, tmp_path):
    from tnlm.data import prepare_data
    from helpers import FixtureTokenizer
    source=tmp_path/'validation.jsonl'
    source.write_text(json.dumps({'id':'validation-only','text':'nopqrstuvwxyz'})+'\n')
    manifest=prepare_data(tmp_path/'validation',source_jsonl=source,revision='canary-fixture',
        tokenizer=FixtureTokenizer(),split_fractions={'train':0.,'val':1.,'test':0.,'calibration':0.})
    config.raw['evaluation']={'validation_manifest':str(manifest)}
    out=tmp_path/'passing'
    result=execute(config,out,canary_steps=2,min_tokens_per_second=.000001,canary_warmup_steps=1)
    assert result['status']=='complete'
    gate=result['stages']['a/parent']['result']['throughput_canary']
    assert gate['status']=='passed' and [x['step'] for x in gate['selected_steps']]==[2,3]
    metrics=[json.loads(line) for line in (out/'a/parent/metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['tokens']==config.train.rung_total_tokens
    assert [x['step'] for x in metrics]==list(range(1,len(metrics)+1))
    assert (out/'a/evaluation/evaluation.json').is_file()
