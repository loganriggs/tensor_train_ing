"""Supervised large-corpus preparation with explicit disposable-raw staging.

Keep the source specification and report outside the immutable output directory.
Only explicitly listed raw files below --prune-raw-root are removed, after the
preparer has verified and completely scanned that shard. Other raw files are kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prune_scanned_shards(source, root, journal, *, append=False):
    """Restrict deletion to declared regular parquet files in the disposable tree."""
    root = Path(root).resolve(strict=True)
    if root in (Path('/'), ROOT, ROOT / 'data', ROOT.parent):
        raise ValueError('disposable raw root must be a dedicated subdirectory')
    selected = []
    for shard in source['shards']:
        original = Path(shard['local'])
        path = original.resolve(strict=True)
        if not path.is_relative_to(root):
            continue
        if original.is_symlink() or not path.is_file() or path.suffix != '.parquet':
            raise ValueError('refusing nonregular disposable raw shard')
        selected.append(path)
    if len(set(selected)) != len(selected):
        raise ValueError('duplicate local raw shard paths')
    record = (json.loads(Path(journal).read_text()) if append and Path(journal).exists() else
              {'phase': 'verified_shards_scanned', 'root': str(root),
               'paths': [], 'removed': [], 'bytes_removed': 0})
    if record['root'] != str(root) or any(str(p) in record['paths'] for p in selected):
        raise ValueError('pruning journal identity or duplicate mismatch')
    record['paths'].extend(str(p) for p in selected)
    Path(journal).write_text(json.dumps(record, indent=2) + '\n')
    for path in selected:
        size = path.stat().st_size
        path.unlink()
        record['removed'].append(str(path))
        record['bytes_removed'] += size
        Path(journal).write_text(json.dumps(record, indent=2) + '\n')
    return record


def worker(args, source):
    from tnlm.data import prepare_data
    declared = {r['path']: r for r in source['shards']}
    def shard_scanned(name, path):
        # Called only after this complete shard's post-scan SHA and reader close.
        if args.prune_raw_root:
            if Path(path).resolve() != Path(declared[name]['local']).resolve():
                raise ValueError('scanned shard path differs from frozen source')
            prune_scanned_shards({'shards': [declared[name]]}, args.prune_raw_root,
                                args.report.with_suffix('.pruning.json'), append=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('preparation stopped by operator')
    signal.signal(signal.SIGTERM, interrupted)
    prepare_data(args.out, hf_dataset=source['dataset'], revision=source['revision'],
        hf_data_files=[r['path'] for r in source['shards']],
        hf_shard_checksums={r['path']: r['sha256'] for r in source['shards']},
        hf_local_shards={r['path']: r['local'] for r in source['shards']},
        max_tokens=args.max_tokens, max_documents=args.max_documents, workers=args.workers,
        exclude_validation_manifest=args.heldout_manifest,
        on_shard_scanned=shard_scanned if args.prune_raw_root else None)



class DiskPauseController:
    """Pause the worker session while this coordinator keeps monitoring free disk."""
    def __init__(self, pgid, pause_bytes, resume_bytes):
        if (type(pause_bytes) is not int or type(resume_bytes) is not int
                or pause_bytes <= 0 or resume_bytes <= pause_bytes):
            raise ValueError('disk resume threshold must exceed a positive pause threshold')
        self.pgid, self.pause_bytes, self.resume_bytes = pgid, pause_bytes, resume_bytes
        self.pauses = []
        self.paused = False

    def _signal(self, signum):
        try:
            os.killpg(self.pgid, signum)
        except ProcessLookupError:
            return False
        return True

    def update(self, free_bytes, elapsed):
        if self.paused:
            self.pauses[-1]['minimum_free_bytes'] = min(self.pauses[-1]['minimum_free_bytes'], free_bytes)
            if free_bytes >= self.resume_bytes:
                self._signal(signal.SIGCONT)
                self.finish(elapsed, 'disk_recovered', free_bytes)
                return True
        elif free_bytes < self.pause_bytes and self._signal(signal.SIGSTOP):
            self.paused = True
            self.pauses.append({'start_elapsed_seconds': elapsed,
                'free_bytes_at_pause': free_bytes, 'minimum_free_bytes': free_bytes,
                'end_elapsed_seconds': None, 'duration_seconds': None,
                'free_bytes_at_resume': None, 'resume_reason': None})
            return True
        return False

    def finish(self, elapsed, reason, free_bytes=None):
        if self.paused:
            row = self.pauses[-1]
            row.update(end_elapsed_seconds=elapsed,
                       duration_seconds=max(0., elapsed-row['start_elapsed_seconds']),
                       free_bytes_at_resume=free_bytes, resume_reason=reason)
            self.paused = False

    def continue_for_shutdown(self, elapsed):
        # Continue even if interrupted between SIGSTOP and state accounting,
        # or an external monitor stopped the group. Stopped Python workers
        # cannot process SIGTERM until continued.
        self._signal(signal.SIGCONT)
        if self.paused:
            self.finish(elapsed, 'operator_or_monitor_shutdown')

    def report(self, elapsed):
        total = sum(row['duration_seconds'] if row['duration_seconds'] is not None
                    else max(0., elapsed-row['start_elapsed_seconds']) for row in self.pauses)
        return {'pause_threshold_bytes': self.pause_bytes, 'resume_threshold_bytes': self.resume_bytes,
                'paused': self.paused, 'pause_seconds': total, 'pauses': self.pauses,
                'scope': 'SIGSTOP/SIGCONT whole worker process group; coordinator keeps monitoring; sampled guard, not reserved disk'}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--heldout-manifest', type=Path, required=True)
    p.add_argument('--prune-raw-root', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=13_000_000_000)
    p.add_argument('--max-documents', type=int, default=30_000_000)
    p.add_argument('--minimum-train-tokens', type=int, default=12_500_000_000)
    p.add_argument('--minimum-free-bytes', type=int, default=35 * 1024**3,
                   help='pause worker process group below this free-space threshold (default35GiB)')
    p.add_argument('--resume-free-bytes', type=int, default=40 * 1024**3,
                   help='resume worker process group at this free-space threshold (default40GiB)')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.minimum_free_bytes <= 0 or args.resume_free_bytes <= args.minimum_free_bytes:
        p.error('resume-free-bytes must exceed positive minimum-free-bytes')
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    if args.worker:
        worker(args, source)
        return 0
    if args.out.exists() or args.report.exists():
        raise FileExistsError('output/report must be fresh')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    import psutil
    frozen_source = args.report.with_suffix('.source.json').resolve()
    with frozen_source.open('xb') as stream:
        stream.write(source_bytes)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv), '--source', str(frozen_source), '--worker']
    started = time.monotonic()
    peak = parent_peak = 0
    free_min = shutil.disk_usage(args.out.parent).free
    guard_error = None
    implementations = {str(path.relative_to(ROOT)): sha(path) for path in
        (ROOT/'tnlm/data.py', ROOT/'tnlm/data_preparation.py', Path(__file__).resolve())}
    with args.report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        process = psutil.Process(child.pid)
        disk_pause = DiskPauseController(child.pid, args.minimum_free_bytes, args.resume_free_bytes)
        def write_pauses():
            args.report.with_suffix('.pauses.json').write_text(
                json.dumps(disk_pause.report(time.monotonic()-started), indent=2)+'\n')
        def interrupted(signum, frame):
            raise KeyboardInterrupt('supervisor requested stop')
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            last_progress = 0
            while child.poll() is None:
                rss = 0
                try:
                    for descendant in [process, *process.children(recursive=True)]:
                        try:
                            size = descendant.memory_info().rss
                            rss += size
                            if descendant.pid == child.pid:
                                parent_peak = max(parent_peak, size)
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                peak = max(peak, rss)
                free = shutil.disk_usage(args.out.parent).free
                free_min = min(free_min, free)
                if disk_pause.update(free, time.monotonic()-started):
                    write_pauses()
                    print(json.dumps({'disk_paused': disk_pause.paused, 'free_bytes': free}), flush=True)
                if time.monotonic() - last_progress > 30:
                    print(json.dumps({'elapsed_seconds': time.monotonic()-started,
                        'free_bytes': free, 'rss_bytes': rss, 'peak_rss_bytes': peak,
                        'disk_paused': disk_pause.paused}), flush=True)
                    last_progress = time.monotonic()
                time.sleep(.25)
        except BaseException as error:
            guard_error = guard_error or str(error)
            disk_pause.continue_for_shutdown(time.monotonic()-started)
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        finally:
            disk_pause.finish(time.monotonic()-started, 'worker_exited')
            write_pauses()
            signal.signal(signal.SIGTERM, previous_handler)
    elapsed = time.monotonic() - started
    report = {'returncode': child.returncode, 'error': guard_error,
              'elapsed_seconds': elapsed, 'disk_pause': disk_pause.report(elapsed),
              'active_elapsed_seconds': max(0., elapsed-disk_pause.report(elapsed)['pause_seconds']),
              'throughput_scope': 'wall-clock preparation time includes disk pauses',
              'peak_process_tree_rss_bytes': peak,
              'peak_parent_rss_bytes': parent_peak, 'minimum_free_bytes_observed': free_min,
              'source_spec_sha256': source_digest, 'frozen_source_spec': str(frozen_source), 'implementation_sha256': implementations,
              'rss_scope': 'sampled summed parent/descendant RSS every0.25s; shared pages counted per process',
              'accepted': False}
    manifest_path = args.out/'manifest.json'
    if child.returncode == 0 and not guard_error:
        manifest = json.loads(manifest_path.read_text())
        report.update(manifest=str(manifest_path.resolve()), manifest_sha256=sha(manifest_path),
            tokens=manifest['tokens'], documents=manifest['documents'],
            tokens_per_second=sum(manifest['tokens'].values()) / elapsed,
            train_tokens_per_second=manifest['tokens']['train'] / elapsed,
            minimum_train_tokens_met=manifest['tokens']['train'] >= args.minimum_train_tokens,
            acceptance_status='awaiting independent artifact and heldout audit')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if child.returncode == 0 and not guard_error and report.get('minimum_train_tokens_met') else 1


if __name__ == '__main__':
    raise SystemExit(main())
