import json
from pathlib import Path
import pytest
import yaml
from tnlm.cli import main


def args(tmp_path):
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump({'project':{'dry_run':True},
        'train':{'global_prediction_tokens_per_update':8192, 'lr':.00045}}))
    return ['branch','--config',str(path),'--resume','parent','--run-dir',str(tmp_path/'out'),
            '--max-steps','4','--max-seconds','30','--device','cpu',
            '--batch-tokens','32768','--lr-scale','2']


def test_cli_branch_forwards_explicit_intervention(tmp_path,monkeypatch,capsys):
    calls=[]
    def train(cfg,run_dir,**kw):
        assert cfg.train.global_prediction_tokens_per_update==8192
        calls.append(kw);return {'branch':True}
    monkeypatch.setattr('tnlm.cli.train_run',train)
    assert main(args(tmp_path))==0
    assert calls[0]['batch_tokens']==32768 and calls[0]['lr_scale']==2.
    assert calls[0]['resume']=='parent'
    assert json.loads(capsys.readouterr().out)=={'branch':True}


def test_actual_dry_branch_does_not_read_parent_or_create_output(tmp_path,capsys):
    assert main(args(tmp_path))==0
    report=json.loads(capsys.readouterr().out)
    assert report['batch_branch']=={'old_batch_tokens':8192,'new_batch_tokens':32768,'lr_scale':2.}
    assert report['parent_validation']=='pending'
    assert not (tmp_path/'out').exists()


def test_distributed_branch_rejected_before_launch(tmp_path):
    with pytest.raises(ValueError,match='serial'):
        main(args(tmp_path)+['--distributed'])
    assert not (tmp_path/'out').exists()
