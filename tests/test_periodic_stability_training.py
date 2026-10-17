"""Periodic eager diagnostics must not perturb the actual training trajectory."""
from pathlib import Path
import json
import pytest
import torch
import yaml
from tnlm.config import Config
from helpers import state
from tnlm.train import train_run


def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('package', ['pre_rms', 'lagged'])
def test_periodic_logging_preserves_training_and_resume(config,tmp_path,package):
    config.model.normalization=package
    config.model.final_normalization='lagged' if package=='lagged' else 'rms'
    config.raw['data']['calibration_tokens']=8
    config.raw['project'].update(keep_recovery_checkpoints=0,save_initial_checkpoint=False,milestone_tokens=[])
    config.train.stability_every_steps=0
    baseline=train_run(config,tmp_path/'baseline',max_steps=4,max_seconds=60,device='cpu')
    config.train.stability_every_steps=2
    yaml_path = tmp_path/'stability.yaml'
    yaml_path.write_text(yaml.safe_dump(config.to_dict()))
    config = Config.from_yaml(yaml_path)
    assert config.train.stability_every_steps == 2
    whole=train_run(config,tmp_path/'whole',max_steps=4,max_seconds=60,device='cpu')
    prefix=train_run(config,tmp_path/'prefix',max_steps=2,max_seconds=60,device='cpu')
    resumed=train_run(config,tmp_path/'resumed',max_steps=2,max_seconds=60,device='cpu',resume=prefix['checkpoint'])
    a=state(Path(baseline['checkpoint']));b=state(Path(whole['checkpoint']));c=state(Path(resumed['checkpoint']))
    for key in ('model','optimizer','sampler','progress'):
        equal(a[key],b[key]);equal(b[key],c[key])
    rows=[json.loads(l) for l in (tmp_path/'whole/metrics.jsonl').read_text().splitlines()]
    logs=[r for r in rows if 'stability' in r]
    assert [r['step'] for r in logs]==[2,4]
    assert logs[0]['stability']['probe']==logs[1]['stability']['probe']
    split=[json.loads(l) for l in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]
    assert split[-1]['stability']['probe']==logs[-1]['stability']['probe']
    assert split[-1]['stability']['stats']==logs[-1]['stability']['stats']
    assert b['config']['train']['stability_every_steps']==2


def test_default_off_does_not_construct_probe(config,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('off must not prepare probe')
    monkeypatch.setattr('tnlm.train.TrainingStabilityProbe',forbidden)
    train_run(config,tmp_path/'off',max_steps=1,max_seconds=60,device='cpu')


@pytest.mark.parametrize('value',[-1,True,1.5,'2'])
def test_invalid_interval_rejects_before_output(config,tmp_path,value):
    config.train.stability_every_steps=value
    with pytest.raises(ValueError):train_run(config,tmp_path/'bad',max_steps=1,max_seconds=60,device='cpu')
    assert not (tmp_path/'bad').exists()
