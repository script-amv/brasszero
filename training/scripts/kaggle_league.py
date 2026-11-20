# SPDX-License-Identifier: Apache-2.0
"""Private Kaggle CPU entrypoint. The local builder pins this job's input digest."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
import time
import traceback
import urllib.request


INPUT_SHA = '__INPUT_SHA256__'
RECOVERY_SHA = '__RECOVERY_SHA256__'


def restore_return(archive, destination):
    """Extract trusted hash-pinned regular files and internal POSIX hardlinks."""
    with tarfile.open(archive) as package:
        members = package.getmembers()
        previous = set()
        for item in members:
            path = Path(item.name)
            if (path.is_absolute() or not path.parts or path.parts[0]!='league-return' or
                    '..' in path.parts or item.name in previous):
                raise ValueError('invalid study recovery archive path')
            if item.islnk():
                if item.linkname not in previous:
                    raise ValueError('invalid study recovery hardlink')
            elif not (item.isfile() or item.isdir()):
                raise ValueError('unsupported entry in recovery archive')
            previous.add(item.name)
        package.extractall(destination)


def arm_identities(work):
    return {arm:json.loads(path.read_bytes())['id'] for arm in ('old','new')
            for path in [work/'study'/arm/'experiment.json'] if path.exists()}


def verify_sealed_prefix(sealed, actual):
    """New local work may extend an immutable return; its committed data must survive."""
    def sha(path):
        with path.open('rb') as stream:
            h = hashlib.sha256()
            for chunk in iter(lambda:stream.read(1048576),b''):
                h.update(chunk)
            return h.hexdigest()
    def journal(path):
        lines = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
        return lines[0], [v for v in lines[1:] if 'status' not in v]
    for path in (sealed/'study').rglob('*'):
        rel = path.relative_to(sealed)
        if not path.is_file():
            continue
        parts = rel.parts
        immutable = (parts[1] in ('protocol.json','arm-identities.json','evaluation') or
                     len(parts)>2 and parts[1] in ('old','new') and parts[2] in
                     ('experiment.json','config.json','policies','weights','updates'))
        target = actual/rel
        if parts[1:3]==('replays','games'):
            if path.suffix=='.jsonl':
                head,moves = journal(path)
                completed = target.with_suffix('.json.gz')
                if completed.exists():
                    with gzip.open(completed) as reader:
                        record = json.load(reader)
                    longer = record['moves']
                elif target.exists():
                    record,longer = journal(target)
                else:
                    raise ValueError('missing sealed prefix: '+str(rel))
                if (longer[:len(moves)]!=moves or any(head[k]!=record[k] for k in
                        ('game_id','initial','seats','engine_id','provenance'))):
                    raise ValueError('divergent sealed prefix: '+str(rel))
                continue
            immutable = True
        if not immutable:
            continue
        if path.name=='optimizer.pt' and not target.exists():
            receipt = target.parent/'retention.json'
            if receipt.exists():
                data = json.loads(receipt.read_bytes())
                if (data.get('retired')=={'optimizer.pt':sha(path)} and
                        data.get('manifest_sha256')==sha(path.parent/'manifest.json')):
                    continue
        if not target.is_file() or sha(target)!=sha(path):
            raise ValueError('missing or changed sealed prefix: '+str(rel))


def restore_provider_input(work, inputs):
    marker = work/'input-provenance.json'
    if RECOVERY_SHA!='fresh':
        descriptor, = inputs.rglob('league-recovery.json')
        if hashlib.sha256(descriptor.read_bytes()).hexdigest()!=RECOVERY_SHA:
            raise ValueError('recovery descriptor pin mismatch')
        recovery = json.loads(descriptor.read_bytes())
        archive = descriptor.parent/'prior-return.bin'
        if (recovery['input_sha256']!=INPUT_SHA or
                hashlib.sha256(archive.read_bytes()).hexdigest()!=recovery['archive_sha256']):
            raise ValueError('recovery data differ from sealed input or prior output')
        fresh = not work.exists() or not any(work.iterdir())
        if fresh:
            restore_return(archive,work.parent)
        if any(arm_identities(work).get(arm)!=identity for arm,identity in recovery['arms'].items()):
            raise ValueError('restored arm identity differs from prior provider output')
        if not fresh:
            with tempfile.TemporaryDirectory(prefix='brass-check-prefix-') as staging:
                restore_return(archive,Path(staging))
                verify_sealed_prefix(Path(staging)/'league-return',work)
    work.mkdir(exist_ok=True)
    if marker.exists():
        if hashlib.sha256(marker.read_bytes()).hexdigest()!=INPUT_SHA:
            raise ValueError('existing working study belongs to a different sealed input')
    elif (work/'study').exists():
        raise ValueError('existing study lacks original provider input provenance')


def execute(scratch, work):
    def command(args, **kwargs):
        subprocess.run([str(a) for a in args],check=True,env=env,**kwargs)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_path, = Path('/kaggle/input').rglob('league-input.json')
    if sha(manifest_path)!=INPUT_SHA:
        raise ValueError('unrecognized Kaggle input version')
    manifest=json.loads(manifest_path.read_bytes())
    source=scratch/'source'
    for name,dest in [('inputs.bin',scratch/'inputs'),('source.bin',source)]:
        path=manifest_path.parent/name
        if sha(path)!=manifest['files'][name]:
            raise ValueError('Kaggle input payload hash mismatch')
        with tarfile.open(path) as tar:
            for item in tar.getmembers():
                if Path(item.name).is_absolute() or '..' in Path(item.name).parts or not (item.isfile() or item.isdir()):
                    raise ValueError('invalid member in compute input')
            tar.extractall(dest)
    shutil.copyfile(manifest_path,work/'input-provenance.json')
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',CARGO_BUILD_JOBS='2',UV_DEFAULT_INDEX='https://pypi.org/simple',
             UV_PYTHON_INSTALL_DIR=str(scratch/'python'),PYTHONUNBUFFERED='1',
             PYTHONPATH=str(source/'engine/tools')+':'+str(source/'training'),
             BRASS_SOURCE_REVISION=manifest['revision'])
    env.pop('UV_INDEX_URL',None)
    uv=shutil.which('uv')
    if not uv:
        command([sys.executable,'-m','pip','install','--target',scratch/'uv','uv'])
        uv=scratch/'uv/bin/uv'
    command([uv,'python','install','3.10'])
    command([uv,'venv','--managed-python','--python','3.10',scratch/'venv'])
    py=scratch/'venv/bin/python'
    command([uv,'pip','install','--python',py,'torch==2.2.2','--index-url','https://download.pytorch.org/whl/cpu'])
    command([uv,'pip','install','--python',py,'-r',source/'training/requirements/linux.txt'])
    env['PATH']=str(Path.home()/'.cargo/bin')+':'+env['PATH']
    if not shutil.which('cargo',path=env['PATH']):
        installer=scratch/'rustup.sh'
        installer.write_bytes(urllib.request.urlopen('https://sh.rustup.rs',timeout=45).read())
        env.update(CARGO_HOME=str(scratch/'cargo'),RUSTUP_HOME=str(scratch/'rustup'))
        env['PATH']=str(scratch/'cargo/bin')+':'+env['PATH']
        command(['sh',installer,'-y','--profile','minimal','--no-modify-path'])
    probe=subprocess.run(['rustc','--version'],env=env,capture_output=True,text=True)
    if probe.returncode and 'no default is configured' in probe.stderr:
        command(['rustup','toolchain','install','stable','--profile','minimal'])
        command(['rustup','default','stable'])
    command([scratch/'venv/bin/maturin','build','--release','--locked','-m',source/'engine/bindings/Cargo.toml',
             '--interpreter',py,'-o',scratch/'wheels'])
    wheel,=(scratch/'wheels').glob('*.whl')
    command([uv,'pip','install','--python',py,'--no-deps',wheel])
    with (work/'runtime.txt').open('w') as out:
        command([uv,'pip','freeze','--python',py],stdout=out)
        command(['lscpu'],stdout=out)
        command([py,'-m','brass_rl','runtime','--device','cpu'],stdout=out)
    process = subprocess.Popen([str(py),'-m','brass_rl.league_study',str(scratch/'inputs'),str(work/'study')],env=env)
    while True:
        try:
            code = process.wait(timeout=60)
            if code:
                raise subprocess.CalledProcessError(code,process.args)
            break
        except subprocess.TimeoutExpired:
            progress = {}
            for arm in ('old','new'):
                checkpoints = work/'study'/arm/'updates'
                progress[arm] = max((int(p.name) for p in checkpoints.glob('[0-9]*') if p.name.isdecimal()),default=None)
                games = work/'study/evaluation'/arm/'games'
                progress[arm+'_evaluated'] = sum(1 for _ in games.glob('*.json'))
            print('PROGRESS',json.dumps(progress),flush=True)


def main():
    work=Path('/kaggle/working/league-return')
    restore_provider_input(work,Path('/kaggle/input'))
    started=time.monotonic()
    status='failed'
    try:
        with tempfile.TemporaryDirectory(prefix='brass-league-') as scratch:
            execute(Path(scratch),work)
        status='complete'
    except Exception:
        (work/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        session = json.dumps(dict(status=status,wall_seconds=time.monotonic()-started,
                entry_python=platform.python_version(),cpu_count=os.cpu_count(),accelerator='none'))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (work/'sessions').mkdir(exist_ok=True)
        (work/'sessions'/(stamp+'.json')).write_text(session)
        (work/'session.json').write_text(session)
        # Hard links between journals and portable versions stay deduplicated.
        with tarfile.open('/kaggle/working/.league-return.tmp','w:gz',compresslevel=1) as tar:
            tar.add(work,arcname='league-return')
        os.replace('/kaggle/working/.league-return.tmp','/kaggle/working/league-return.tar.gz')


if __name__=='__main__':
    main()
