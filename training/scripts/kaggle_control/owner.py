# SPDX-License-Identifier: Apache-2.0
"""Owner commands. Credentials are loaded only from the private Mac file."""
import argparse
import errno
import re
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time

from common import control, digest, extract, read, write
from campaign import lock as owner_lock, route_stop


def show(value):
    print(json.dumps(value, sort_keys=True),flush=True)


def kaggle(*args, capture=False):
    executable = shutil.which('kaggle') or str(Path.home()/'.local/bin/kaggle')
    return subprocess.run([executable,*map(str,args)],check=True,text=True,capture_output=capture)


def validate_prefix(old, new):
    # Every historical model, journal, replay and session stays byte-identical.
    # A prior sealed latest is pinned for retention before it leaves Kaggle.
    for sub in ('run/updates','run/policies','run/weights','replays/games','sessions'):
        for path in (old/sub).rglob('*'):
            if path.is_file():
                other = new/path.relative_to(old)
                if not other.is_file() or digest(path) != digest(other):
                    raise ValueError('changed prior state file '+str(path.relative_to(old)))
    for name in ('experiment.json','config.json'):
        if digest(old/'run'/name) != digest(new/'run'/name):
            raise ValueError('experiment or configuration changed')


def save_stop(directory, result, sent, confirmed, purpose='operator'):
    path = directory/'stop-receipt.json'
    with owner_lock(directory):
        if not path.exists():
            write(path,dict(run=result['run'],requested=True,requested_at=sent,confirmed_at=confirmed,purpose=purpose))
        return read(path)


def remember(directory, job, remote):
    phases = ['bootstrap','training','ack_stopping','checkpoint_saved','exporting','export_ready']
    def rank(row):
        return (-1,-1) if not row else (phases.index(row['phase']),row['seq'])
    path = directory/'last-status.json'
    with owner_lock(directory):
        prior = read(path) if path.exists() else {}
        progress = max([prior.get('progress'),remote.get('progress')],key=rank)
        result = dict(run=job['run'],requested=bool(remote['requested'] or prior.get('requested')
                      or (directory/'stop-receipt.json').exists()),progress=progress)
        write(path,result)
        return result


def stop_target(directory, job):
    # Linearize an owner stop with publication of the proof's single successor.
    with owner_lock(directory.parent):
        active = route_stop(directory,job)
        if active is not None:
            return active
        if job.get('stage') != 'proof':
            return directory
        with owner_lock(directory):
            halt = directory/'operator-halt.json'
            if not halt.exists():
                write(halt,dict(at=time.time(),run=job['run']))
            successor = directory/'successor.json'
            if not successor.exists():
                return directory
            chosen = read(successor)
            target = directory.parent / Path(chosen['directory']).name
            next_job = read(target/'job.json')
            if next_job['run'] != chosen['run'] or next_job['proof_run'] != job['run']:
                raise ValueError('campaign successor identity mismatch')
            return target


def provider_status(kernel):
    status = kaggle('kernels','status',kernel,capture=True).stdout.strip()
    value = re.fullmatch(re.escape(kernel) + r' has status "(?:KernelWorkerStatus[.])?([A-Za-z_]+)"',status.splitlines()[0],re.I)
    if not value:
        raise ValueError('unrecognized Kaggle status: '+status)
    return value.group(1).upper()


def download(directory, job, *, project=None, local_only=False):
    import os
    from train_run import verify
    dest = directory/'verified'
    if local_only and not dest.is_dir():
        raise ValueError('sealed local return is missing')
    with tempfile.TemporaryDirectory(prefix='download-',dir=None if local_only else directory) as tmp:
        stage = Path(tmp)/'package'
        stage.mkdir()
        if dest.exists():
            for name in ('return.json','continuation.tar.gz'):
                try:
                    os.link(dest/name,stage/name)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(dest/name,stage/name)
        else:
            if provider_status(job['kernel']) != 'COMPLETE':
                raise ValueError('pinned Kaggle version has not completed publication')
            kaggle('kernels','output',job['kernel'],'-p',stage,'--file-pattern','^(continuation[.]tar[.]gz|return[.]json)$')
        receipt = read(stage/'return.json')
        archive = stage/'continuation.tar.gz'
        if (receipt['deadline'] != job['deadline'] or digest(archive) != receipt['sha256'] or receipt['run'] != job['run'] or
                receipt['revision'] != job['revision'] or receipt['experiment'] != job['initial']['experiment'] or
                receipt['initial_archive'] != job['initial']['sha256']):
            raise ValueError('wrong return archive, source, or run')
        if dest.is_dir() and read(dest/'verified.json').get('sha256') != receipt['sha256']:
            raise ValueError('sealed archive differs from local verification')
        if receipt.get('initial_manifest') != job['initial']['manifest']:
            raise ValueError('return seed manifest differs from prepared job')
        extract(archive,stage/'.unpacked')
        (stage/'unpacked').symlink_to('.unpacked', target_is_directory=True)
        state = stage/'unpacked/state'
        initial_archive = directory/'dataset/state.bin'
        if digest(initial_archive) != receipt['initial_archive']:
            raise ValueError('stored starting archive changed')
        extract(initial_archive,Path(tmp)/'initial')
        validate_prefix(Path(tmp)/'initial/state',state)
        summary = verify(state)
        if summary['manifest'] != receipt['manifest'] or summary['update'] != receipt['update']:
            raise ValueError('returned tip differs from provider receipt')
        summary['sha256'] = receipt['sha256']
        summary.update(control_id=job['run'],kernel=job['kernel'],revision=job['revision'],
                       deadline=job['deadline'],reason=receipt['reason'],ack_at=receipt.get('ack_at'),
                       outcome=receipt['outcome'],requested_at=receipt.get('requested_at'))
        if dest.exists():
            if read(dest/'verified.json') != summary:
                raise ValueError('stale local verification receipt differs from pinned return')
            if not local_only:
                validate_prefix(state,dest/'unpacked/state')
        else:
            write(stage/'verified.json',summary)
            os.rename(stage,dest)
            state = dest/'unpacked/state'
        if project is not None:
            from brass_rl.library import _publish_kaggle_state
            return _publish_kaggle_state(state, project, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('command', choices=['status','stop','proof-stop','download','watch','ingest'])
    parser.add_argument('--wait-download', action='store_true')
    parser.add_argument('--project-root', type=Path, help='destination data project for local library ingestion')
    args = parser.parse_args()
    directory = args.directory.resolve()
    job = read(directory/'job.json')
    if args.command == 'stop':
        directory = stop_target(directory,job)
        job = read(directory/'job.json')
    project = args.project_root or job.get('library_project')
    if (args.command in ('download', 'ingest') or args.wait_download) and not project:
        parser.error('set --project-root to the destination data project')
    if args.command in ('download', 'ingest'):
        show(download(directory, job, project=Path(project), local_only=args.command == 'ingest'))
        return
    key = Path.home()/'.local/share/brass-kaggle-control'/job['run']/'owner.json'
    if key.is_symlink() or stat.S_IMODE(key.stat().st_mode) != 0o600:
        raise ValueError('owner capability must be a mode-0600 file')
    cap = read(key)
    if cap['run'] != job['run']:
        raise ValueError('wrong owner capability')
    if args.command in ('stop','proof-stop'):
        testing = args.command == 'proof-stop'
        if testing and job.get('stage') != 'proof':
            raise ValueError('proof-stop is restricted to the proof session')
        intent = directory/'stop-intent.json'
        with owner_lock(directory):
            halt = directory/'operator-halt.json'
            if testing and halt.exists():
                raise ValueError('operator already ended this campaign')
            if not testing and not halt.exists():
                write(halt,dict(at=time.time(),run=job['run']))
            if not intent.exists():
                write(intent,dict(requested_at=int(time.time()*1000)))
            payload = read(intent)
        result = control(cap,'stop',payload)
        show(save_stop(directory,result,payload['requested_at']/1000,time.time(),'proof' if testing else 'operator'))
        if not args.wait_download:
            return
    while True:
        try:
            show(remember(directory,job,control(cap,'status')))
        except OSError:
            print('Control status unavailable; checking Kaggle publication.',flush=True)
        status = provider_status(job['kernel'])
        print(status,flush=True)
        if status == 'COMPLETE':
            if args.wait_download:
                show(download(directory, job, project=Path(project)))
            return
        if status in ('ERROR','CANCELLED','CANCELED','CANCEL_REQUESTED','CANCEL_ACKNOWLEDGED','FAILED'):
            raise RuntimeError('provider ended without a confirmed normal publication')
        if args.command == 'status':
            return
        time.sleep(90)


if __name__ == '__main__':
    main()
