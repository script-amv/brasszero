# SPDX-License-Identifier: Apache-2.0
"""Kaggle lifecycle. Local stop markers survive all subsequent polling errors."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

from common import WHEEL_NAME, WHEEL_SHA, control, digest, extract, read, seal, write

GIB = 1024**3
RESERVE = 30*60
UPDATE_MARGIN = 10*60


class Monitor:
    def __init__(self, cap, initial, deadline, state, signals, output):
        self.cap, self.update, self.deadline = cap, initial, deadline
        self.state, self.signals, self.output = state, signals, output
        self.phase, self.reason, self.seq = 'bootstrap', 'none', 0
        self.confirmed = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.last_contact = time.monotonic()
        self.last_poll = -float('inf')
        self.last_sent = -float('inf')
        self.last_phase = None
        self.ack_at = None
        self.requested_at = None

    def stop(self, reason):
        with self.lock:
            if self.reason != 'none':
                return
            (self.signals/'stop').touch()
            self.reason = reason
            self.ack_at = int(time.time()*1000)
            if self.phase in ('bootstrap','training'):
                self.phase = 'ack_stopping'
        self.wake.set()

    def advance(self, phase):
        with self.lock:
            if phase == 'training' and self.reason != 'none':
                return
            self.phase = phase
        self.wake.set()

    def status(self):
        with self.lock:
            return dict(phase=self.phase,reason=self.reason,update=self.update,seq=self.seq,ack_at=self.ack_at)

    def observe(self):
        tip = self.state/'run/latest'
        if tip.exists():
            self.update = int(read(tip)['update'])
        if (self.signals/'active').exists() and self.phase == 'bootstrap':
            self.advance('training')
        if time.monotonic() >= self.deadline:
            self.stop('deadline')
        elif time.monotonic()-self.last_contact > 180:
            self.stop('control_unavailable')
        # The scratch filesystem and Kaggle's output volume can differ.
        used = sum(p.stat().st_size for p in self.state.rglob('*') if p.is_file())
        if (used >= 8*GIB or shutil.disk_usage(self.state).free < GIB or
                shutil.disk_usage(self.output).free < used + GIB):
            self.stop('capacity')

    def tick(self):
        now = time.monotonic()
        if now-self.last_poll >= 35:
            self.last_poll = now
            try:
                row = control(self.cap, 'poll')
                if type(row.get('requested')) is not bool:
                    raise ValueError('invalid stop state')
                if row['requested']:
                    if self.requested_at is None:
                        self.requested_at = row.get('requested_at')
                    self.stop('manual')
                self.last_contact = time.monotonic()
                self.confirmed.set()
            except (OSError, ValueError):
                print('control poll unavailable', flush=True)
        row = self.status()
        if row['phase'] != self.last_phase or now-self.last_sent >= 120:
            self.seq += 1
            row['seq'] = self.seq
            # Count attempts even when the server accepted a write and its response was lost.
            self.last_phase, self.last_sent = row['phase'], now
            try:
                control(self.cap, 'progress', row)
            except (OSError, ValueError):
                print('control status unavailable', flush=True)
                return
            print('CONTROL', row, flush=True)

    def loop(self):
        while not self.finished.is_set():
            try:
                self.observe()
            except FileNotFoundError:
                # Atomic caches and short-lived trainer snapshots can disappear mid-scan.
                pass
            except Exception:
                self.stop('control_unavailable')
            try:
                self.tick()
            except Exception:
                # A malformed transport reply cannot strand the numerical child.
                self.stop('control_unavailable')
            (self.signals/'heartbeat').touch()
            self.wake.wait(5)
            self.wake.clear()


def setup(scratch, source, env, monitor, wheel_input):
    class StopSetup(Exception):
        pass
    def command(args, **kwargs):
        if (monitor.signals/'stop').exists():
            raise StopSetup
        # Avoid interrupting a healthy setup subprocess. Its termination is unrelated to training.
        subprocess.run([str(a) for a in args], check=True, env=env,
                       timeout=max(5,min(360,monitor.deadline-time.monotonic())), **kwargs)
    uv = shutil.which('uv')
    py = scratch/'venv/bin/python'
    try:
        if not uv:
            command([sys.executable,'-m','pip','install','--target',scratch/'uv','uv'])
            uv = scratch/'uv/bin/uv'
        command([uv,'python','install','3.10'])
        command([uv,'venv','--managed-python','--python','3.10',scratch/'venv'])
        if digest(wheel_input) != WHEEL_SHA:
            raise ValueError('unexpected Linux CPU wheel bytes')
        wheel_path = scratch/WHEEL_NAME
        shutil.copyfile(wheel_input,wheel_path)
        command([uv,'pip','install','--python',py,wheel_path,'-r',source/'training/requirements/linux.txt'])
        env['PATH'] = str(Path.home()/'.cargo/bin')+':'+env['PATH']
        if not shutil.which('cargo',path=env['PATH']):
            script = scratch/'rustup.sh'
            with urllib.request.urlopen('https://sh.rustup.rs',timeout=25) as stream:
                script.write_bytes(stream.read())
            env.update(CARGO_HOME=str(scratch/'cargo'), RUSTUP_HOME=str(scratch/'rustup'))
            env['PATH'] = str(scratch/'cargo/bin')+':'+env['PATH']
            command(['sh',script,'-y','--profile','minimal','--no-modify-path'])
        probe = subprocess.run(['rustc','--version'],env=env,capture_output=True,text=True)
        if probe.returncode and 'no default is configured' in probe.stderr:
            command(['rustup','toolchain','install','stable','--profile','minimal'])
            command(['rustup','default','stable'])
        command([scratch/'venv/bin/maturin','build','--release','--locked','-m',
                source/'engine/bindings/Cargo.toml','--interpreter',py,'-o',scratch/'wheels'])
        wheel, = (scratch/'wheels').glob('*.whl')
        command([uv,'pip','install','--python',py,'--no-deps',wheel])
        with (monitor.state/'runtime.txt').open('w') as stream:
            command([uv,'pip','freeze','--python',py],stdout=stream)
            command([py,'-m','brass_rl','runtime','--device','cpu'],stdout=stream)
        return py
    except StopSetup:
        return None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        if time.monotonic() >= monitor.deadline:
            monitor.stop('deadline')
        if (monitor.signals/'stop').exists():
            return None
        raise


def run(manifest_path, scratch, started, output):
    m = read(manifest_path)
    data = manifest_path.parent
    for name in ('state.bin','runner.json'):
        if digest(data/name) != m['files'][name]:
            raise ValueError('private input hash mismatch')
    if 'control_endpoint' in m:
        os.environ['BRASS_CONTROL_ENDPOINT'] = m['control_endpoint']
    cap = read(data/'runner.json')
    if cap['run'] != m['run']:
        raise ValueError('wrong control capability run binding')
    # Kaggle publishes this tree as fallback when an archive cannot be completed.
    # Runtime setup, source code and capabilities stay in separate scratch/input paths.
    live = output/'continuation-live'
    live.mkdir(exist_ok=False)
    extract(data/'state.bin',live)
    state = live/'state'
    signals = scratch/'signals'
    signals.mkdir()
    initial = m['initial']
    if (read(state/'run/experiment.json')['id'] != initial['experiment'] or
            digest(state/'run/updates'/('%06d'%initial['update'])/'manifest.json') != initial['manifest']):
        raise ValueError('sealed starting experiment/checkpoint mismatch')
    provider_end = min(m['deadline'], m.get('session_deadline',m['deadline']), started+12*3600)
    monitor = Monitor(cap, initial['update'], time.monotonic()+provider_end-time.time()-RESERVE-UPDATE_MARGIN,
                      state,signals,output)
    thread = threading.Thread(target=monitor.loop,daemon=True)
    thread.start()
    sealed = False
    failure = None
    try:
        source = scratch/'src'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CARGO_BUILD_JOBS='1',
                   UV_DEFAULT_INDEX='https://pypi.org/simple',UV_PYTHON_INSTALL_DIR=str(scratch/'python'),
                   PYTHONUNBUFFERED='1',BRASS_SOURCE_REVISION=m['revision'],BRASS_REPLAY_ROOT=str(state/'replays'),
                   PYTHONPATH=os.pathsep.join(map(str,[source/'engine/tools',source/'training'])))
        env.pop('UV_INDEX_URL',None)
        env['BRASS_TRAIN_CUTOFF'] = str(monitor.deadline)
        while not monitor.confirmed.wait(1) and not (signals/'stop').exists():
            pass
        try:
            if m['files'].get('torch-cpu.bin') != WHEEL_SHA and not (signals/'stop').exists():
                raise ValueError('CPU wheel descriptor mismatch')
            py = setup(scratch,source,env,monitor,data/'torch-cpu.bin')
        except (OSError,subprocess.SubprocessError):
            failure = 'setup_error'
            monitor.stop(failure)
            py = None
        if py and not (signals/'stop').exists():
            entry = source/'training/scripts/kaggle_control/train_run.py'
            result = subprocess.run([py,entry,'train',state,signals],env=env)
            if result.returncode:
                failure = 'trainer_error'
                monitor.stop(failure)
                # The writer exited; restore/repair only committed bytes, without collecting a game.
                subprocess.run([py,entry,'recover',state,m['run']],env=env,check=True)
            elif monitor.reason == 'none':
                notice = signals/'child-stop.json'
                monitor.stop(read(notice)['reason'] if notice.exists() else 'trainer_ended')
        current = int(read(state/'run/latest')['update'])
        boundary = state/'run/updates'/('%06d'%current)
        if not (boundary/'optimizer.pt').exists() or not (state/'run/policies'/boundary.name/'manifest.json').exists():
            raise ValueError('full latest state or export missing')
        monitor.update = current
        # Phase records have separate keys. Flush these quick transitions in order.
        monitor.finished.set()
        monitor.wake.set()
        thread.join()
        def phase(name):
            monitor.advance(name)
            monitor.tick()
        if monitor.reason != 'none':
            phase('ack_stopping')
        phase('checkpoint_saved')
        pin = state/'run/updates/pins'/(digest(boundary/'manifest.json')+'.json')
        pin.parent.mkdir(exist_ok=True)
        if not pin.exists():
            from json import dumps
            pin.write_text(dumps({'manifest_sha256':digest(boundary/'manifest.json'),'update':current},
                                 separators=(',',':'),sort_keys=True)+'\n')
        phase('exporting')
        receipt = dict(run=m['run'],revision=m['revision'],experiment=initial['experiment'],update=current,
                initial_manifest=initial['manifest'],manifest=digest(boundary/'manifest.json'),
                initial_archive=m['files']['state.bin'],deadline=m['deadline'],started=started,
                reason=monitor.reason,ack_at=monitor.ack_at,requested_at=monitor.requested_at,
                outcome='recovered' if failure else 'normal',failure=failure,session_deadline=provider_end)
        sessions = state/'sessions'
        sessions.mkdir(exist_ok=True)
        write(sessions/(m['run']+'.json'),receipt)
        receipt['sha256'] = seal(state,output/'continuation.tar.gz')
        write(output/'return.json',receipt)
        sealed = True
        phase('export_ready')
    except Exception as exc:
        # Preserve the whole workspace if committed recovery or final publication fails.
        # A separate name prevents operators from treating an unchecked tree as a normal return.
        recovery = output/'recovery-evidence.tar.gz'
        evidence = dict(run=m['run'],error=type(exc).__name__)
        try:
            evidence['sha256'] = seal(state,recovery)
            sealed = True
        except OSError:
            evidence['directory'] = str(live.name)
        write(output/'recovery-required.json',evidence)
        print('Recovery evidence published; inspection required.',flush=True)
    finally:
        monitor.finished.set()
        monitor.wake.set()
        thread.join()
        if sealed:
            shutil.rmtree(live)


if __name__ == '__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]),float(sys.argv[3]),Path('/kaggle/working'))
