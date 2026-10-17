"""Actual CPU/gloo cosine replay across the warmup boundary."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import FixtureTokenizer
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data, RankPackedSampler
from tnlm.distributed_train import distributed_train_run
from tnlm.distributed_checkpoint import load_distributed_checkpoint
from tnlm.model import TensorLM
from tnlm.parallel import shard_model, full_state_dict
from tnlm.train import make_optimizer
from tnlm.evaluate import _training_descriptor


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=150))
    rank=dist.get_rank();root=Path(sys.argv[1]);mesh=init_device_mesh('cpu',(2,))
    if rank==0:
        source=root/'source.jsonl';source.write_text(json.dumps({'id':'long','text':'abcdefgh'*200})+'\n')
        prepare_data(root/'data',source_jsonl=source,revision='offline-cosine',max_tokens=1000,
            tokenizer=FixtureTokenizer(),split_fractions={'train':1.,'val':0.,'test':0.,'calibration':0.})
    dist.barrier()
    manifest=root/'data/manifest.json'
    cfg=Config(model=ModelConfig(layers=1,d_model=8,heads=1,d_key=4,d_value=4,d_mlp=12,
        vocab_size=128,padded_vocab=128,attention_backend='reference'),
        train=TrainConfig(schedule='cosine',compile=False,bf16_compute=False,lr=.01,
            rung_total_tokens=713,global_prediction_tokens_per_update=7),
        raw={'data':{'manifest':str(manifest),'context':4,'micro_batch_size':1},
             'project':{'dry_run':False,'gpu_hourly_rate_usd':0.,'spend_ledger':str(root/'spend.json'),
                        'milestone_tokens':[],'checkpoint_every_seconds':1000000}})
    whole=distributed_train_run(cfg,root/'whole',max_steps=102,max_seconds=120)
    first=distributed_train_run(cfg,root/'first',max_steps=100,max_seconds=120)
    resumed=distributed_train_run(cfg,root/'resumed',max_steps=2,max_seconds=120,resume=first['checkpoint'])
    equal(whole['progress'],resumed['progress'])
    assert whole['progress']['tokens']==713 and whole['progress']['scheduler_step']==102
    assert whole['history'][99]['optimizer_group_lrs'][0]==.01
    assert abs(whole['history'][100]['optimizer_group_lrs'][0]-.005)<1e-15
    assert whole['history'][101]['optimizer_group_lrs'][0]==0
    assert whole['history'][-1]['prediction_tokens']==6
    assert whole['history'][100:]==resumed['history'] or all(
        a['rounds']==b['rounds'] and a['optimizer_group_lrs']==b['optimizer_group_lrs']
        for a,b in zip(whole['history'][100:],resumed['history']))
    loaded=[]
    for result in (whole,resumed):
        model=shard_model(TensorLM(cfg.model),mesh);optimizer=make_optimizer(model,cfg.train)
        sampler=RankPackedSampler(manifest,rank=rank,world_size=2,context=4,seed=cfg.train.seed)
        payload=load_distributed_checkpoint(result['checkpoint'],model,optimizer,sampler)
        loaded.append((full_state_dict(model),get_optimizer_state_dict(model,optimizer,
            options=StateDictOptions(full_state_dict=True,cpu_offload=True)),sampler.state_dict()))
        descriptor=_training_descriptor({'config':payload['config'],'progress':payload['progress']})
        assert descriptor['schedule']=='cosine' and descriptor['completion_status']=='complete'
    equal(loaded[0],loaded[1])
    if rank==0:
        for name in ('whole','first','resumed'):
            assert not list((root/name).glob('branchpoint-*'))
            assert not list((root/name).glob('milestone-*'))
    before=(root/'spend.json').read_bytes()
    for name,promote,kwargs in [('promotion',True,{}),('cooldown',False,{'cooldown_steps':2})]:
        bad=copy.deepcopy(cfg)
        if promote:bad.train.rung_total_tokens+=1
        try:distributed_train_run(bad,root/name,max_steps=1,max_seconds=120,resume=first['checkpoint'],**kwargs)
        except ValueError:pass
        else:raise AssertionError(name+' accepted')
        assert not (root/name).exists()
    assert (root/'spend.json').read_bytes()==before
    if rank==0:print('cosine exact warmup-boundary replay and prebilling rejection passed',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
