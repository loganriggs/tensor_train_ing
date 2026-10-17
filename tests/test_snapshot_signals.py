"""Real process-group cancellation; no trainer or GPU is started."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tnlm.source_snapshot import _run_child


def wait_file(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None, 'supervisor exited before child became ready'
        assert time.monotonic() < deadline, 'child startup timed out'
        time.sleep(.02)


def running(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize('ignore', [False, True])
def test_cancellation_reaches_child_and_descendant(tmp_path, signum, ignore):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
role = sys.argv[2]
ignore = sys.argv[3] == 'True'
def stop(signum, frame):
    (root / (role + '.signal')).write_text(str(signum))
    if not ignore:
        raise SystemExit(0)
for s in (signal.SIGTERM, signal.SIGINT):
    signal.signal(s, stop)
if role == 'child':
    subprocess.Popen([sys.executable, __file__, str(root), 'grandchild', str(ignore)])
(root / (role + '.pid')).write_text(str(os.getpid()))
while True:
    time.sleep(.05)
''')
    launcher = '''import sys
from tnlm.source_snapshot import _run_child
raise SystemExit(_run_child(sys.argv[1:], cwd=None, shutdown_seconds=.4))
'''
    proc = subprocess.Popen([sys.executable, '-c', launcher, sys.executable,
                             str(worker), str(tmp_path), 'child', str(ignore)])
    pids = []
    try:
        for role in ('child', 'grandchild'):
            wait_file(tmp_path / (role + '.pid'), proc)
            pids.append(int((tmp_path / (role + '.pid')).read_text()))
        proc.send_signal(signum)
        assert proc.wait(timeout=10) == 128 + signum
        deadline = time.monotonic() + 3
        while any(running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not any(running(pid) for pid in pids)
        # An ignoring group must receive the original signal before forced kill.
        if ignore:
            for role in ('child', 'grandchild'):
                assert (tmp_path / (role + '.signal')).read_text() == str(signum)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_normal_exit_and_spawn_failure_restore_handlers(tmp_path):
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    assert _run_child([sys.executable, '-c', 'raise SystemExit(7)'], cwd=tmp_path) == 7
    assert {s: signal.getsignal(s) for s in previous} == previous
    with pytest.raises(FileNotFoundError):
        _run_child([str(tmp_path / 'absent')], cwd=tmp_path)
    assert {s: signal.getsignal(s) for s in previous} == previous
