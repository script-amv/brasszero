# SPDX-License-Identifier: Apache-2.0
"""Detached, single-writer local training supervisor (stdlib only).

Control files live beside the run, allowing the trainer's exclusive empty-root
check to remain effective. Signals are sent only to a child owned by this process.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import select
import subprocess
import sys
import time

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class Limits:
    hours: float = 8
    soft_bytes: int = 1536*MIB
    hard_bytes: int = 2*GIB
    soft_free: int = 2*GIB
    hard_free: int = 512*MIB
    grace: float = 1800

    def __post_init__(self):
        if not math.isfinite(self.hours) or not 0 < self.hours <= 8:
            raise ValueError('hours must be positive and at most 8')


def control_dir(out):
    return out.with_name(out.name+'-control')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


@contextmanager
def writer_lock(control):
    with (control/'lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('training supervisor is active')
        try:
            yield stream
        finally:
            pass  # Closing releases our descriptor; inherited child retains its lock.


def stop_reason(elapsed, size, free, limits):
    if elapsed >= limits.hours*3600:
        return 'hard', 'deadline'
    if size >= limits.hard_bytes:
        return 'hard', 'disk'
    if free <= limits.hard_free:
        return 'hard', 'free_disk'
    if size >= limits.soft_bytes:
        return 'graceful', 'disk'
    if free <= limits.soft_free:
        return 'graceful', 'free_disk'
    if elapsed >= max(0, limits.hours*3600-limits.grace):
        return 'graceful', 'time'
    return None, None


def disk_bytes(*roots):
    total = 0
    for root in roots:
        if root.exists():
            for parent, dirs, files in os.walk(root, followlinks=False):
                for name in files:
                    try:
                        total += (Path(parent)/name).lstat().st_size
                    except FileNotFoundError:
                        pass
    return total


def caffeinate(pid):
    if sys.platform == 'darwin':
        return subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None


def kill_owned(child, number):
    if child.poll() is None:
        try:
            os.killpg(child.pid, number)
        except ProcessLookupError:
            pass


def monitor(child, out, control, limits, poll=2, clock=time.monotonic, caffeinate=caffeinate, started=None):
    started = clock() if started is None else started; stopping = None; reason = None; power = None
    try:
        power = caffeinate(child.pid)
        while child.poll() is None:
            elapsed = clock()-started
            size = disk_bytes(out, control)
            free = shutil.disk_usage(control).free
            action, why = stop_reason(elapsed, size, free, limits)
            if stopping is not None and clock()-stopping >= limits.grace:
                action, why = 'hard', reason
            if (control/'stop').exists() and action != 'hard':
                action, why = 'graceful', 'manual'
            if action == 'hard':
                reason = why
                kill_owned(child, signal.SIGKILL)
            elif action == 'graceful' and stopping is None:
                reason = why; stopping = clock()
                kill_owned(child, signal.SIGTERM)
            atomic_json(control/'status.json', {'state': 'stopping' if stopping is not None else 'running',
                        'pid': child.pid, 'reason': reason, 'elapsed_seconds': elapsed,
                        'bytes': size, 'free_bytes': free, 'updated_at': time.time()})
            if action == 'hard':
                break
            remaining = limits.hours*3600-(clock()-started)
            if stopping is not None:
                remaining = min(remaining, limits.grace-(clock()-stopping))
            time.sleep(max(0, min(poll, remaining)))
        code = child.wait()
        atomic_json(control/'status.json', {'state': 'finished' if code == 0 else 'stopped' if reason else 'failed',
                    'returncode': code, 'reason': reason, 'elapsed_seconds': clock()-started,
                    'updated_at': time.time()})
    finally:
        kill_owned(child, signal.SIGKILL)
        child.wait()
        if power is not None:
            if power.poll() is None:
                power.terminate()
            power.wait()


def file_limit():
    resource.setrlimit(resource.RLIMIT_FSIZE, (64*MIB, 64*MIB))


def supervise(out, resume, lock_fd, ready_fd):
    control = control_dir(out)
    started = time.monotonic()
    child = None
    descriptors = set()
    def interrupted(signum, frame):
        raise RuntimeError("supervisor interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Inherited locked descriptor keeps the reservation uninterrupted.
        metadata = json.loads((control/'metadata.json').read_text())
        file_limit()
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        child_read, child_ready = os.pipe()
        descriptors.update((child_read, child_ready))
        life_read, life_write = os.pipe()
        descriptors.update((life_read, life_write))
        launcher = """import os, signal, threading
# Keep the flock alive in the actual writer. Only the supervisor holds the
# pipe writer; EOF means all disk/deadline supervision has been lost.
os.set_inheritable(%d, False)
os.set_inheritable(%d, False)
def supervisor_lost():
    try:
        while os.read(%d, 1):
            pass
    finally:
        os.killpg(os.getpgrp(), signal.SIGKILL)
threading.Thread(target=supervisor_lost, daemon=True).start()
import torch
from contextlib import contextmanager
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from brass_rl import __main__ as entry
original = entry._graceful_signals
@contextmanager
def ready(stop):
    with original(stop):
        os.write(%d, b'OK')
        os.close(%d)
        yield
entry._graceful_signals = ready
entry.main()
""" % (lock_fd, life_read, life_read, child_ready, child_ready)
        command = [sys.executable, '-c', launcher, 'train', '--config', str(control/'config.json'),
                   '--out', str(out), '--max-updates', str(metadata['max_updates'])]
        if resume:
            command += ['--resume', 'latest']
        with (control/'training.log').open('ab') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True, pass_fds=(child_ready, life_read, lock_fd), preexec_fn=lambda: os.nice(10))
            os.close(child_ready); descriptors.remove(child_ready)
            os.close(life_read); descriptors.remove(life_read)
            readable, _, _ = select.select([child_read], [], [], 12)
            ready = os.read(child_read, 2) if readable else b''
            os.close(child_read); descriptors.remove(child_read)
            if ready != b'OK' or child.poll() is not None:
                raise RuntimeError('trainer exited during startup; inspect '+str(control/'training.log'))
            atomic_json(control/'status.json', {'state': 'running', 'pid': child.pid,
                        'updated_at': time.time()})
            os.write(ready_fd, b'OK\n'); os.close(ready_fd); ready_fd = None
            monitor(child, out, control, Limits(hours=metadata['hours']), started=started)
    except BaseException as exc:
        # Diagnostics cannot prevent termination or replace the original failure.
        try:
            atomic_json(control/'status.json', {'state': 'failed', 'error': str(exc), 'updated_at': time.time()})
        except Exception:
            pass
        if ready_fd is not None:
            try:
                os.write(ready_fd, ('ERROR '+str(exc)+'\n').encode())
            except OSError:
                pass
        raise
    finally:
        # A second signal must not interrupt cleanup before releasing the lock.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if child is not None:
                kill_owned(child, signal.SIGKILL)
                child.wait()
        finally:
            for descriptor in descriptors | ({ready_fd} if ready_fd is not None else set()):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            os.close(lock_fd)


def validate_committed(out):
    """Check the authoritative latest envelope without importing Torch/pickles.

    The training loader performs full tensor/controller/reference validation.
    Never silently fall back past a damaged numeric committed directory.
    """
    updates = out/'updates'
    candidates = sorted((p for p in updates.iterdir() if p.is_dir() and p.name.isdecimal()),
                        key=lambda p: int(p.name)) if updates.is_dir() else []
    if not candidates:
        raise ValueError('resume requires a committed update')
    latest = candidates[-1]
    manifest = json.loads((latest/'manifest.json').read_text())
    from .retention import present_payloads
    if manifest.get('update') != int(latest.name) or 'optimizer.pt' not in present_payloads(latest, manifest):
        raise ValueError('invalid committed update envelope')
    for name, digest in manifest['content_hashes'].items():
        if hashlib.sha256((latest/name).read_bytes()).hexdigest() != digest:
            raise ValueError('committed update hash mismatch: '+name)


def launch(out, config, resume=False, hours=8, max_updates=200):
    out = out.resolve()
    limits = Limits(hours=hours)
    if max_updates <= 0:
        raise ValueError('max-updates must be positive')
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('start requires a new or empty output directory')
    control = control_dir(out)
    control.mkdir(parents=True, exist_ok=True)
    with writer_lock(control) as lock:
        if resume:
            validate_committed(out)
            metadata = json.loads((control/'metadata.json').read_text())
            if hashlib.sha256((control/'config.json').read_bytes()).hexdigest() != metadata['config_sha256']:
                raise ValueError('immutable control config hash mismatch')
            max_updates = metadata['max_updates']
            if not (control/'config.json').is_file():
                raise ValueError('immutable control config is missing')
        else:
            if (control/'metadata.json').exists():
                raise ValueError('control directory already belongs to a run; choose another --out')
            from .config import TrainConfig
            parsed = TrainConfig.from_json(config)
            if parsed.runtime.torch_threads != 1:
                raise ValueError("bounded training requires runtime.torch_threads=1")
            (control/'config.json').write_bytes(config.read_bytes())
            try:
                revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            metadata = {'source_revision': revision, 'resolved_python': str(Path(sys.executable).resolve()), 'max_updates': max_updates, 'out': str(out), 'python': sys.executable,
                        'cwd': os.getcwd(), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'source_config': str(config.absolute()), 'created_at': time.time()}
        metadata['hours'] = limits.hours
        atomic_json(control/'metadata.json', metadata)
        (control/'stop').unlink(missing_ok=True)
        atomic_json(control/'status.json', {'state': 'starting', 'updated_at': time.time()})
        read_fd, write_fd = os.pipe()
        with (control/'supervisor.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-m', 'brass_rl.bounded', '_supervise',
                        '--out', str(out), '--lock-fd', str(lock.fileno()), '--ready-fd', str(write_fd)]
                        + (['--resume'] if resume else []), start_new_session=True,
                        pass_fds=(lock.fileno(), write_fd), stdout=log, stderr=subprocess.STDOUT)
        os.close(write_fd)
        # Do not unlock the shared open-file description when releasing parent handle.
        readable, _, _ = select.select([read_fd], [], [], 15)
        message = os.read(read_fd, 4096).decode() if readable else ''
        os.close(read_fd)
        if not message.startswith('OK'):
            process.terminate(); process.wait(timeout=10)
            raise RuntimeError(message.strip() or 'supervisor startup timed out; inspect '+str(control/'supervisor.log'))
    return json.loads((control/'status.json').read_text())


def status(out):
    """Report only atomic status and committed JSON, never unpickle or load Torch."""
    out = out.resolve()
    control = control_dir(out)
    result = json.loads((control/'status.json').read_text())
    try:
        with writer_lock(control):
            result['active'] = False
    except RuntimeError:
        result['active'] = True
    if not result['active'] and result['state'] in ('running', 'starting', 'stopping'):
        result['state'] = 'interrupted'
    result['paths'] = {'training_log': str(control/'training.log'),
                       'metrics_cache': str(out/'metrics.jsonl'), 'latest': str(out/'latest')}
    updates = out/'updates'
    candidates = sorted((p for p in updates.iterdir() if p.is_dir() and p.name.isdecimal()),
                        key=lambda p: int(p.name)) if updates.is_dir() else []
    if candidates:
        latest = candidates[-1]
        manifest = json.loads((latest/'manifest.json').read_text())
        raw = (latest/'metrics.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['content_hashes']['metrics.json']:
            raise ValueError('committed metrics hash mismatch')
        records = json.loads(raw)['records']
        result['completed_updates'] = manifest['update']
        result['latest_committed'] = str(latest)
        result['latest_metrics'] = records[-1] if records else None
        result['next_game_id'] = records[-1]['next_game_id'] if records else 0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'resume', 'status', 'stop', '_supervise'):
        item = sub.add_parser(name); item.add_argument('--out', type=Path, required=True)
        if name in ('start', 'resume'):
            item.add_argument('--hours', type=float, default=8)
            item.add_argument('--max-updates', type=int, default=200)
        if name == 'start':
            item.add_argument('--config', type=Path, required=True)
        if name == '_supervise':
            item.add_argument('--lock-fd', type=int, required=True)
            item.add_argument('--ready-fd', type=int, required=True)
            item.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == '_supervise':
            supervise(args.out, args.resume, args.lock_fd, args.ready_fd); return
        if args.command in ('start', 'resume'):
            result = launch(args.out, getattr(args, 'config', None), args.command == 'resume', args.hours, args.max_updates)
        elif args.command == 'stop':
            control = control_dir(args.out.resolve())
            # No PID reads/signals. A stopped run's marker is harmless and resume clears it.
            if not (control/'metadata.json').exists():
                raise ValueError('no bounded run at this output')
            (control/'stop').touch()
            result = {'stop_requested': True}
        else:
            result = status(args.out)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
