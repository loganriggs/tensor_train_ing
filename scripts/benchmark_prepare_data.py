"""Reproduce pinned-shard preparation and sample aggregate process-tree RSS.

Requires psutil for measurement. Source JSON contains dataset/revision and shards
with path, sha256, local. Raw shard acquisition is deliberately outside the timed
region. Report and controls stay outside the immutable dataset directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def heldout_audit(manifest, heldout):
    """Stream new metadata; hold only the small existing validation identity set."""
    old, new = (json.loads(Path(p).read_text()) for p in (heldout, manifest))
    ids, raw, processed = set(), set(), set()
    for path, info in ((heldout, old), (manifest, new)):
        assert sha(Path(path).with_name('documents.jsonl')) == info['artifacts']['documents.jsonl']
    with Path(heldout).with_name('documents.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['split'] == 'val':
                ids.add(row['id'])
                raw.add(row['content_sha256'])
                processed.add(row['processed_sha256'])
    overlaps = dict.fromkeys(('train', 'calibration', 'test'), 0)
    examples = []
    with Path(manifest).with_name('documents.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['split'] not in overlaps:
                continue
            if row['id'] in ids or row['content_sha256'] in raw or row['processed_sha256'] in processed:
                overlaps[row['split']] += 1
                if len(examples) < 10:
                    examples.append({'id': row['id'], 'split': row['split']})
    return {'heldout_manifest_sha256': sha(heldout), 'validation_documents': len(ids),
            'overlaps': overlaps, 'examples': examples,
            'scope': 'all existing validation IDs and exact raw/processed full-document hashes; not near-duplicate detection'}


def main():
    import psutil
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--heldout-manifest', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=3_100_000_000)
    p.add_argument('--max-documents', type=int, default=10_000_000)
    args = p.parse_args()
    source = json.loads(Path(args.source).read_text())
    source_sha = sha(args.source)
    repo = Path(__file__).resolve().parents[1]
    implementation = {name: sha(repo / name) for name in
                      ('tnlm/data.py', 'tnlm/data_preparation.py', 'tnlm/cli.py',
                       'scripts/benchmark_prepare_data.py')}
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        raise FileExistsError(report)
    controls = report.with_suffix('.controls')
    controls.mkdir()
    for name, key in (('checksums', 'sha256'), ('local', 'local')):
        (controls / (name + '.json')).write_text(json.dumps({s['path']: s[key] for s in source['shards']}))
    command = [sys.executable, '-m', 'tnlm.cli', 'prepare_data', '--hf-dataset', source['dataset'],
               '--revision', source['revision'], '--hf-data-files', *[s['path'] for s in source['shards']],
               '--hf-shard-checksums', str(controls / 'checksums.json'),
               '--hf-local-shards', str(controls / 'local.json'), '--out', args.out,
               '--workers', str(args.workers), '--max-tokens', str(args.max_tokens),
               '--max-documents', str(args.max_documents),
               '--exclude-validation-manifest', args.heldout_manifest]
    started = time.monotonic()
    peak = root_peak = count_peak = samples = 0
    with report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        parent = psutil.Process(child.pid)
        while child.poll() is None:
            rss = 0
            try:
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                try:
                    value = process.memory_info().rss
                    rss += value
                    if process.pid == child.pid:
                        root_peak = max(root_peak, value)
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            count_peak = max(count_peak, len(processes))
            samples += 1
            time.sleep(.1)
    elapsed = time.monotonic() - started
    result = {'command': command, 'returncode': child.returncode, 'elapsed_seconds': elapsed,
              'peak_process_tree_rss_bytes': peak, 'peak_parent_rss_bytes': root_peak,
              'peak_process_count': count_peak, 'rss_samples': samples,
              'rss_scope': 'sum of parent and descendant RSS sampled every0.1s; shared pages counted per process; short peaks may be missed',
              'throughput_scope': 'whole preparation including local raw hashing, selection, tokenization, artifact hashing/fsync; excludes acquisition and post-run audit',
              'source_spec_sha256': source_sha, 'workers': args.workers,
              'implementation_sha256': implementation, 'python': platform.python_version(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('tiktoken', 'numpy', 'pyarrow', 'torch', 'psutil')},
              'cpu_affinity_count': len(os.sched_getaffinity(0)),
              'thread_environment': {name: os.environ.get(name) for name in
                                     ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}}
    if child.returncode == 0:
        manifest = Path(args.out) / 'manifest.json'
        data = json.loads(manifest.read_text())
        result.update(manifest=str(manifest.resolve()), manifest_sha256=sha(manifest),
                      tokens=data['tokens'], documents=data['documents'],
                      tokens_per_second=sum(data['tokens'].values()) / elapsed,
                      train_tokens_per_second=data['tokens']['train'] / elapsed,
                      heldout_audit=heldout_audit(manifest, args.heldout_manifest))
        result['accepted'] = not any(result['heldout_audit']['overlaps'].values())
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return child.returncode or (0 if result.get('accepted') else 2)


if __name__ == '__main__':
    raise SystemExit(main())
