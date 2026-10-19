import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('prepare_scaling_data', Path(__file__).resolve().parents[1]/'scripts/prepare_scaling_data.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_pruning_only_explicit_disposable_shards(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir()
    disposable=raw/'one.parquet'; disposable.write_bytes(b'123')
    unrelated=raw/'keep.parquet'; unrelated.write_bytes(b'456')
    old=tmp_path/'old.parquet'; old.write_bytes(b'789')
    source={'shards':[{'local':str(disposable)}, {'local':str(old)}]}
    report=m.prune_scanned_shards(source, raw, tmp_path/'pruning.json')
    assert not disposable.exists()
    assert old.exists() and unrelated.exists()
    assert report['bytes_removed']==3
    assert json.loads((tmp_path/'pruning.json').read_text())['removed']==[str(disposable)]


def test_symlinks_and_duplicate_paths_reject_before_deletion(tmp_path):
    path=tmp_path/'one.parquet'; path.write_bytes(b'123')
    alias=tmp_path/'alias.parquet'; alias.symlink_to(path)
    with pytest.raises(ValueError, match='nonregular'):
        m.prune_scanned_shards({'shards':[{'local':str(path)}, {'local':str(alias)}]}, tmp_path, tmp_path/'journal')
    assert path.exists()
    with pytest.raises(ValueError, match='duplicate'):
        m.prune_scanned_shards({'shards':[{'local':str(path)}]*2}, tmp_path, tmp_path/'journal')
    assert path.exists()


def test_worker_prunes_only_after_successful_scan_and_publishes(tmp_path):
    import hashlib
    import os
    import subprocess
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tnlm.data import prepare_data
    heldout_source=tmp_path/'heldout.jsonl'
    heldout_source.write_text(json.dumps({'id':'held-out', 'text':'Reserved validation document.'})+'\n')
    heldout=prepare_data(tmp_path/'heldout', source_jsonl=heldout_source, revision='fixture-v1',
        split_fractions={'train':0.,'val':1.,'test':0.,'calibration':0.})
    raw=tmp_path/'raw'; raw.mkdir()
    shard=raw/'fixture.parquet'
    pq.write_table(pa.table({'id':['held-out']+[f'train-{i}' for i in range(20)],
        'text':['Reserved validation document.']+['Fresh training document words.']*20}), shard)
    source=tmp_path/'source.json'; source.write_text(json.dumps({'dataset':'fixture/local', 'revision':'a'*40,
        'shards':[{'path':'fixture.parquet','local':str(shard),'sha256':hashlib.sha256(shard.read_bytes()).hexdigest()}]}))
    report=tmp_path/'report.json'; output=tmp_path/'output'
    command=[sys.executable,str(Path(m.__file__)), '--source',str(source), '--out',str(output),
        '--report',str(report), '--heldout-manifest',str(heldout), '--prune-raw-root',str(raw),
        '--workers','2','--max-tokens','1000','--max-documents','100','--minimum-train-tokens','1',
        '--minimum-free-bytes','1']
    run=subprocess.run(command,capture_output=True,text=True,timeout=90,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'))
    assert run.returncode==0,run.stdout+run.stderr+report.with_suffix('.log').read_text()
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['heldout_exclusion']['excluded_unique_documents']==1
    assert manifest['tokens']['train']>0
    assert not shard.exists()
    measured=json.loads(report.read_text())
    assert measured['accepted'] is False
    assert measured['minimum_train_tokens_met'] is True
