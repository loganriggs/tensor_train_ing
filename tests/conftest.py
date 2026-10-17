"""Shared CPU integration fixtures; test modules do not import one another."""
import json

import pytest
import torch

from tnlm.checkpoint import save_checkpoint
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data, manifest_hash
from tnlm.tasks import prepare_mechanism_tasks
from tnlm.model import TensorLM
from tnlm.train import full_config
from helpers import FixtureTokenizer


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefghijklm'}) + '\n' for i in range(20)))
    manifest = prepare_data(tmp_path / 'data', source_jsonl=source, revision='offline-v1',
                            tokenizer=FixtureTokenizer(),
                            split_fractions={'train': .8, 'val': 0., 'test': 0., 'calibration': .2})
    return Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                   d_mlp=12, padded_vocab=50257),
                  train=TrainConfig(compile=False, bf16_compute=False,
                                    global_prediction_tokens_per_update=7, rung_total_tokens=100, lr=.01),
                  raw={'data': {'manifest': str(manifest), 'context': 4, 'micro_batch_size': 1, 'calibration_tokens': 8},
                       'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0,
                                   'spend_ledger': str(tmp_path / 'spend.json')}})



@pytest.fixture
def checkpoint_fixture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'id': 'training-source', 'text': 'offline language data'}) + '\n')
    training = prepare_data(tmp_path / 'data', source_jsonl=source, revision='offline-v1',
        split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    tasks = prepare_mechanism_tasks(tmp_path / 'tasks', seed=5, train_examples=8, validation_examples=8)
    cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                   d_mlp=12, padded_vocab=50257),
                 train=TrainConfig(compile=False, bf16_compute=False),
                 raw={'data': {'manifest': str(training), 'context': 128},
                      'project': {'dry_run': False, 'gpu_hourly_rate_usd': .01,
                                  'spend_ledger': str(tmp_path / 'spend.json')}})
    model = TensorLM(cfg.model)
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(checkpoint, model, None, None, config=full_config(cfg), progress={'steps': 0},
                    manifest_hash=manifest_hash(training))
    return cfg, checkpoint, tasks




def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: real training/sweep, compilation, and multiprocess integration checks; exclude with -m "not slow" during quick local iterations')
