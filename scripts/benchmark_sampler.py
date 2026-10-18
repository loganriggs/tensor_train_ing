#!/usr/bin/env python3
"""Measure fresh-process sampler startup/RSS and reproducible batch lineage."""
import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy-source', help='Frozen tnlm/data.py for a before measurement')
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--build-index-only', action='store_true')
    parser.add_argument('--verify-long-resume', action='store_true', help='Scan to near epoch end and verify exact restore')
    args = parser.parse_args()
    if args.batches < 1:
        parser.error('--batches must be positive')
    import torch
    from tnlm.data import batch_hash, manifest_hash
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location('tnlm._sampler_baseline', args.legacy_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.PackedSampler
        source = Path(args.legacy_source)
    else:
        from tnlm.data import PackedSampler
        cls = PackedSampler
        source = Path(__file__).resolve().parents[1] / 'tnlm/data.py'
    torch.set_num_threads(1)
    if args.build_index_only:
        from tnlm.sampler_index import ensure_sampler_index
        manifest_path = Path(args.manifest)
        started = time.perf_counter()
        ensure_sampler_index(manifest_path.parent, json.loads(manifest_path.read_text()))
        result = dict(index_build_seconds=time.perf_counter()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            manifest_sha256=manifest_hash(manifest_path))
        Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started = time.perf_counter()
    sampler = cls(args.manifest, split='train', batch_size=16, context=512, seed=0)
    initialized = time.perf_counter()
    first = sampler.next_batch()
    first_done = time.perf_counter()
    first_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    hashes = [batch_hash(first)]
    lineage = [hashlib.sha256(json.dumps(sampler.batch_mappings(), sort_keys=True).encode()).hexdigest()]
    for _ in range(args.batches - 1):
        hashes.append(batch_hash(sampler.next_batch()))
        lineage.append(hashlib.sha256(json.dumps(sampler.batch_mappings(), sort_keys=True).encode()).hexdigest())
    state = sampler.state_dict()
    meminfo = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Shared_Clean', 'Shared_Dirty'):
            meminfo[key + '_bytes'] = int(value.split()[0]) * 1024
    result = dict(implementation='legacy' if args.legacy_source else 'compact',
        manifest_sha256=manifest_hash(args.manifest), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initialization_seconds=initialized-started, first_batch_seconds=first_done-initialized,
        startup_and_first_batch_seconds=first_done-started, peak_rss_bytes=first_peak,
        imported_process_peak_rss_bytes=baseline_peak, memory_after_batches=meminfo,
        training_documents=len(sampler.docs), batch_sha256=hashes, lineage_sha256=lineage,
        state=state, scope='Fresh CPU process, B16/T512/seed0; artifact verification included; imports excluded from time but included in RSS')
    if args.verify_long_resume:
        scan_start = time.perf_counter()
        row_cursor = max(0, sampler.epoch_row_count - 2 * sampler.batch_size)
        document_cursor, token_cursor = sampler._seek_row(sampler.order, row_cursor)
        near_end = {**state, 'cursor': row_cursor, 'document_cursor': document_cursor,
                    'token_cursor': token_cursor}
        sampler.load_state_dict(near_end)
        expected = batch_hash(sampler.next_batch())
        expected_mappings = sampler.batch_mappings()
        sampler.load_state_dict(near_end)
        assert batch_hash(sampler.next_batch()) == expected
        assert sampler.batch_mappings() == expected_mappings
        result['long_resume'] = dict(seconds=time.perf_counter()-scan_start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            checkpoint=near_end, batch_sha256=expected, exact_replay=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
