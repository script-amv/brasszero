# SPDX-License-Identifier: Apache-2.0
"""Fast lifecycle tests; no numerical runtime required."""
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pytest
from brass_rl import bounded as b


def test_policy_absolute_deadline_and_disk():
    p = b.Limits(hours=8)
    assert b.stop_reason(27000, 0, 10*b.GIB, p) == ('graceful', 'time')
    assert b.stop_reason(28800, 0, 10*b.GIB, p) == ('hard', 'deadline')
    assert b.stop_reason(0, 2*b.GIB, 10*b.GIB, p) == ('hard', 'disk')
    assert b.stop_reason(0, 0, 500*b.MIB, p) == ('hard', 'free_disk')


def test_lock_is_exclusive(tmp_path):
    with b.writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match='active'):
            with b.writer_lock(tmp_path):
                pass


def test_supervise_stop_owned_process_and_caffeinate(tmp_path):
    out = tmp_path/'run'
    control = b.control_dir(out)
    control.mkdir()
    b.atomic_json(control/'metadata.json', {'hours': 1, 'max_updates': 2})
    (control/'stop').touch()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    calls = []
    b.monitor(child, out, control, b.Limits(hours=1), poll=.01,
              caffeinate=lambda pid: calls.append(pid))
    assert child.poll() is not None
    assert calls == [child.pid]
    assert json.loads((control/'status.json').read_text())['reason'] == 'manual'


def test_supervise_failure_and_hard_deadline(tmp_path):
    control = tmp_path/'control'; control.mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'], start_new_session=True)
    b.monitor(child, tmp_path/'run', control, b.Limits(hours=1), poll=.01)
    assert json.loads((control/'status.json').read_text())['state'] == 'failed'
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    ticks = iter([0, 10, 10, 10])
    b.monitor(child, tmp_path/'run', control, b.Limits(hours=.000001), poll=.01, clock=lambda: next(ticks))
    assert child.returncode < 0
    assert json.loads((control/'status.json').read_text())['reason'] == 'deadline'


def test_start_rejects_existing_and_resume_missing(tmp_path):
    out = tmp_path/'run'; out.mkdir(); (out/'data').touch()
    with pytest.raises(ValueError, match='empty'):
        b.launch(out, tmp_path/'config', False, 8, 200)
    with pytest.raises((ValueError, OSError)):
        b.launch(out, None, True, 8, 200)


def fake_runtime(tmp_path, monkeypatch, broken=False):
    package = tmp_path/'runtime'/'brass_rl'; package.mkdir(parents=True)
    source = Path(b.__file__).parent
    for name in ('bounded.py', 'config.py', '__init__.py'):
        (package/name).write_bytes((source/name).read_bytes())
    (package.parent/'torch.py').write_text('raise RuntimeError("broken runtime")' if broken else
        'def set_num_threads(n): pass\ndef set_num_interop_threads(n): pass\n')
    (package/'__main__.py').write_text('''import argparse, signal, time
from pathlib import Path
from contextlib import contextmanager
@contextmanager
def _graceful_signals(stop):
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    yield
def main():
    p = argparse.ArgumentParser(); p.add_argument('train'); p.add_argument('--config'); p.add_argument('--out'); p.add_argument('--max-updates'); p.add_argument('--resume'); a=p.parse_args()
    out=Path(a.out); out.mkdir(exist_ok=True)
    if not a.resume: assert not list(out.iterdir())
    (out/('resumed' if a.resume else 'started')).touch()
    with _graceful_signals(None):
        while True: time.sleep(.01)
''')
    monkeypatch.setenv('PYTHONPATH', str(package.parent))
    config = tmp_path/'config.json'; config.write_text('{}')
    return config


def test_detached_launch_keeps_lock_and_stop_marker(tmp_path, monkeypatch):
    config = fake_runtime(tmp_path, monkeypatch)
    out = tmp_path/'run'
    result = b.launch(out, config)
    control = b.control_dir(out)
    assert result['state'] == 'running'
    try:
        (control/'stop').touch()
        # The live supervisor can legitimately refresh status during this call.
        # Check that the rejected launcher itself never writes control state.
        with monkeypatch.context() as blocked:
            def unexpected_write(*args, **kwargs):
                raise AssertionError('rejected launcher wrote control state')
            blocked.setattr(b, 'atomic_json', unexpected_write)
            with pytest.raises(RuntimeError, match='active'):
                b.launch(out, None, resume=True)
        assert (control/'stop').exists()
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            status = json.loads((control/'status.json').read_text())
            if status['state'] == 'finished': break
            time.sleep(.05)
        assert status['state'] == 'finished'
        with b.writer_lock(control): pass
        committed = out/'updates'/'000000'; committed.mkdir(parents=True)
        hashes = {}
        for name in ('policy.pt', 'metadata.json', 'controller.pt', 'optimizer.pt', 'metrics.json', 'manifests.json', 'versions.json'):
            (committed/name).write_bytes(b'{}')
            hashes[name] = hashlib.sha256(b'{}').hexdigest()
        b.atomic_json(committed/'manifest.json', {'format': 'brass-complete-update-v1', 'update': 0, 'content_hashes': hashes})
        resumed = b.launch(out, None, resume=True, hours=1)
        assert resumed['state'] == 'running'
        assert (out/'resumed').exists()
        assert not (control/'stop').exists()
        (control/'stop').touch()
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            if json.loads((control/'status.json').read_text())['state'] == 'finished': break
            time.sleep(.05)
        with b.writer_lock(control): pass
    finally:
        (control/'stop').touch()


def test_startup_failure_is_actionable(tmp_path, monkeypatch):
    config = fake_runtime(tmp_path, monkeypatch, broken=True)
    out = tmp_path/'run'
    with pytest.raises(RuntimeError, match='training.log'):
        b.launch(out, config)
    control = b.control_dir(out)
    assert json.loads((control/'status.json').read_text())['state'] == 'failed'
    with b.writer_lock(control): pass


def test_caffeinate_scope_and_cleanup(tmp_path, monkeypatch):
    calls = []
    class Power:
        def poll(self): return None
        def terminate(self): calls.append('terminate')
        def wait(self): calls.append('wait')
    real = subprocess.Popen
    def spawn(argv, **kwargs):
        calls.append(argv)
        return Power()
    monkeypatch.setattr(b.sys, 'platform', 'darwin')
    monkeypatch.setattr(b.subprocess, 'Popen', spawn)
    power = b.caffeinate(123)
    assert calls == [['/usr/bin/caffeinate', '-i', '-w', '123']]
    monkeypatch.setattr(b.subprocess, 'Popen', real)
    control = tmp_path/'control'; control.mkdir()
    child = real([sys.executable, '-c', 'pass'], start_new_session=True)
    b.monitor(child, tmp_path/'run', control, b.Limits(), poll=.01, caffeinate=lambda pid: power)
    assert calls[-2:] == ['terminate', 'wait']


def test_disk_hard_kills_owned_child(tmp_path, monkeypatch):
    control = tmp_path/'control'; control.mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    monkeypatch.setattr(b, 'disk_bytes', lambda *args: 2*b.GIB)
    b.monitor(child, tmp_path/'run', control, b.Limits(), poll=.01)
    assert child.returncode < 0
    assert json.loads((control/'status.json').read_text())['reason'] == 'disk'


def test_status_detects_dead_supervisor_without_runtime(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ('torch', 'numpy'):
            raise AssertionError('status loaded numerical runtime')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    out = tmp_path/'run'; control = b.control_dir(out); control.mkdir()
    b.atomic_json(control/'status.json', {'state': 'running', 'pid': 1})
    report = b.status(out)
    assert report['state'] == 'interrupted'
    assert report['active'] is False
    assert report['paths']['latest'] == str(out/'latest')


def direct_supervisor(tmp_path, monkeypatch, close_ready=False, fail_status=False):
    import os
    config = fake_runtime(tmp_path, monkeypatch)
    package = tmp_path/'runtime'/'brass_rl'
    main = package/'__main__.py'
    main.write_text(main.read_text().replace("(out/('resumed' if a.resume else 'started')).touch()", "(out/('resumed' if a.resume else 'started')).touch()\n    (out/'pid').write_text(str(__import__('os').getpid()))"))
    out = tmp_path/'run'; control = b.control_dir(out); control.mkdir()
    (control/'config.json').write_bytes(config.read_bytes())
    b.atomic_json(control/'metadata.json', {'hours': 8, 'max_updates': 2})
    lock = os.open(str(control/'lock'), os.O_CREAT | os.O_RDWR, 0o600)
    b.fcntl.flock(lock, b.fcntl.LOCK_EX)
    reader, writer = os.pipe()
    prefix = 'from brass_rl import bounded as b; '
    if fail_status:
        prefix += "b.atomic_json=lambda *args: (_ for _ in ()).throw(OSError('status unavailable')); "
    script = prefix+'from pathlib import Path; b.supervise(Path(%r), False, %d, %d)' % (str(out), lock, writer)
    process = subprocess.Popen([sys.executable, '-c', script], pass_fds=(lock, writer),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(lock); os.close(writer)
    if close_ready:
        os.close(reader); reader = None
    return process, reader, out, control


def assert_child_gone(out):
    import os
    deadline = time.monotonic()+5
    while not (out/'pid').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    pid = int((out/'pid').read_text())
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.01)
    # Keep a failed regression from leaving a runaway fake trainer.
    os.killpg(pid, 9)
    pytest.fail('owned trainer survived supervisor failure')


def test_closed_parent_pipe_cannot_orphan_training(tmp_path, monkeypatch):
    process, reader, out, control = direct_supervisor(tmp_path, monkeypatch, close_ready=True)
    process.wait(timeout=8)
    assert_child_gone(out)
    with b.writer_lock(control): pass


def test_error_status_write_cannot_skip_child_cleanup(tmp_path, monkeypatch):
    import os
    process, reader, out, control = direct_supervisor(tmp_path, monkeypatch, fail_status=True)
    try:
        process.wait(timeout=8)
        assert_child_gone(out)
        with b.writer_lock(control): pass
    finally:
        os.close(reader)


def test_supervisor_sigkill_cannot_leave_unbounded_writer(tmp_path, monkeypatch):
    import os
    process, reader, out, control = direct_supervisor(tmp_path, monkeypatch)
    try:
        assert os.read(reader, 3) == b'OK\n'
        pid = int((out/'pid').read_text())
        os.killpg(pid, b.signal.SIGSTOP)
        process.kill(); process.wait(timeout=5)
        # Even while the watchdog is paused, the actual writer retains flock.
        with pytest.raises(RuntimeError, match='active'):
            with b.writer_lock(control): pass
        os.killpg(pid, b.signal.SIGCONT)
        assert_child_gone(out)
        with b.writer_lock(control): pass
    finally:
        os.close(reader)
        if process.poll() is None: process.kill(); process.wait()
        if (out/'pid').exists():
            try: os.killpg(int((out/'pid').read_text()), 9)
            except ProcessLookupError: pass
