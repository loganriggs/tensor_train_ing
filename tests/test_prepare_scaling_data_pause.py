"""Disk pressure pauses preparation instead of discarding staged work."""
import importlib.util
from pathlib import Path
import signal

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1]/'scripts/prepare_scaling_data.py'
    spec = importlib.util.spec_from_file_location('preparation_disk_pause', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disk_pause_hysteresis_preserves_work_and_reports_durations(monkeypatch):
    module = load_module()
    signals = []
    monkeypatch.setattr(module.os, 'killpg', lambda pgid, sig: signals.append((pgid, sig)))
    guard = module.DiskPauseController(1234, 35, 40)
    assert not guard.update(35, 0.)
    assert guard.update(34, 1.) and guard.paused
    assert not guard.update(25, 2.)
    assert not guard.update(39, 4.) and guard.paused
    assert guard.update(40, 6.) and not guard.paused
    assert signals == [(1234, signal.SIGSTOP), (1234, signal.SIGCONT)]
    report = guard.report(8.)
    assert report['pause_seconds'] == 5.
    assert report['pauses'][0]['minimum_free_bytes'] == 25
    assert report['pauses'][0]['free_bytes_at_resume'] == 40
    assert report['pauses'][0]['resume_reason'] == 'disk_recovered'
    assert guard.update(30, 10.)
    assert guard.report(12.)['pause_seconds'] == 7.


def test_stopped_workers_continued_before_termination(monkeypatch):
    module = load_module()
    signals = []
    monkeypatch.setattr(module.os, 'killpg', lambda pgid, sig: signals.append(sig))
    guard = module.DiskPauseController(1234, 35, 40)
    guard.update(30, 1.)
    guard.continue_for_shutdown(3.)
    module.os.killpg(1234, signal.SIGTERM)
    assert signals == [signal.SIGSTOP, signal.SIGCONT, signal.SIGTERM]
    assert not guard.paused
    assert guard.report(4.)['pause_seconds'] == 2.
    guard.continue_for_shutdown(4.)
    assert signals[-1] == signal.SIGCONT and len(signals) == 4


def test_exited_group_does_not_create_false_pause(monkeypatch):
    module = load_module()
    def gone(*args):
        raise ProcessLookupError('exited')
    monkeypatch.setattr(module.os, 'killpg', gone)
    guard = module.DiskPauseController(1234, 35, 40)
    assert not guard.update(20, 1.)
    assert guard.report(2.)['pauses'] == []


@pytest.mark.parametrize('low,high', [(0, 40), (40, 40), (41, 40)])
def test_pause_threshold_validation(low, high):
    with pytest.raises(ValueError, match='threshold'):
        load_module().DiskPauseController(1234, low, high)


def test_real_worker_group_pauses_and_resumes_without_resetting_progress(tmp_path):
    import os
    import subprocess
    import sys
    import time
    import psutil
    module = load_module()
    worker = """from pathlib import Path
import sys,time
path=Path(sys.argv[1])
count=0
while True:
 count += 1
 path.write_text(str(count))
 time.sleep(.02)
"""
    parent_code = ("import subprocess,sys\nfrom pathlib import Path\n"
        + f"worker={worker!r}\n"
        + "root=Path(sys.argv[1]).parent\n"
        + "child=subprocess.Popen([sys.executable,'-c',worker,str(root/'child.counter')])\n"
        + "(root/'child.pid').write_text(str(child.pid))\nexec(worker)\n")
    parent_counter = tmp_path/'parent.counter'
    child = subprocess.Popen([sys.executable, '-c', parent_code, str(parent_counter)], start_new_session=True)
    guard = module.DiskPauseController(child.pid, 35, 40)
    def until(predicate):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError('worker group did not reach expected state')
    def counter(path):
        try:
            return int(path.read_text())
        except (FileNotFoundError, ValueError):
            return 0
    child_counter = tmp_path/'child.counter'
    try:
        until(lambda: counter(parent_counter) >= 3 and counter(child_counter) >= 3)
        before = (counter(parent_counter), counter(child_counter))
        descendant_pid = int((tmp_path/'child.pid').read_text())
        guard.update(30, 1.)
        until(lambda: all(psutil.Process(pid).status() == psutil.STATUS_STOPPED
                          for pid in (child.pid, descendant_pid)))
        stopped = (parent_counter.read_bytes(), child_counter.read_bytes())
        time.sleep(.15)
        assert (parent_counter.read_bytes(), child_counter.read_bytes()) == stopped
        assert child.poll() is None
        guard.update(40, 2.)
        until(lambda: counter(parent_counter) > before[0] and counter(child_counter) > before[1])
        assert not guard.paused and guard.report(2.)['pause_seconds'] == 1.
    finally:
        guard.continue_for_shutdown(3.)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
