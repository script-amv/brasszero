# SPDX-License-Identifier: Apache-2.0
"""Prepare sealed local inputs and per-run capabilities. This command makes no network requests."""
import argparse
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from common import BASE, trusted_endpoint, POLICY, SOURCE, WHEEL_NAME, WHEEL_SHA, digest, read, seal, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def seed(project, output):
    from brass_rl.artifacts import load_update
    from brass_rl.learning_study import fork_arm
    from brass_rl.train import train
    from train_run import verify
    parent = project/'training/runs/league-study-2026-11-20/old'
    original = parent/'updates/000843'
    if digest(original/'manifest.json') != SOURCE or read(original/'metadata.json')['id'] != POLICY:
        raise ValueError('original checkpoint pin mismatch')
    with tempfile.TemporaryDirectory() as temp:
        scratch = Path(temp)
        state = scratch/'state'
        state.mkdir()
        # fork_arm pins its input. Fork an isolated copy of the exact source bytes.
        copy = scratch/'parent'
        source = copy/'updates/000843'
        shutil.copytree(original, source)
        for name in ('experiment.json','config.json'):
            shutil.copyfile(parent/name, copy/name)
        (copy/'weights').mkdir()
        for snapshot in read(source/'manifest.json')['snapshots']:
            name = snapshot['sha256']+'.pt'
            shutil.copyfile(parent/'weights'/name, copy/'weights'/name)
        from brass_rl.runtime import runtime_check
        runtime_check('cpu')
        loaded = load_update(source)
        cfg = loaded.state.config
        steps = [int(v['step']) for v in loaded.optimizer_state['state'].values()]
        if steps != [18729]*53 or loaded.state.next_game_id != 1686 or loaded.state.development_best_id is not None:
            raise ValueError('starting controller or Adam differs from chosen 843')
        run = state/'run'
        fork_arm(source, run, cfg.ppo.gae_lambda, study_config=cfg)
        os.environ['BRASS_REPLAY_ROOT'] = str(state/'replays')
        report = train(cfg, run, run/'latest', max_updates=BASE)
        if report.completed_updates != BASE:
            raise ValueError('unexpected preflight update')
        for name in ('policy.pt','metadata.json','manifest.json'):
            if digest(run/'policies/000843'/name) != digest(parent/'policies/000843'/name):
                raise ValueError('inherited public export bytes differ')
        summary = verify(state)
        summary['sha256'] = seal(state, output)
        write(output.with_suffix('.json'), summary)


def seed_bc(source, output, *, max_wall_time_seconds=None, experimental=False):
    """Seal a new supported BC child from an explicitly selected full boundary."""
    import torch
    from brass_rl.bc_current import fork_bc
    from brass_rl.experiments import writer_lock
    from train_run import verify

    output = Path(output)
    summary_path = output.with_suffix('.json')
    with writer_lock(output):
        if summary_path == output or output.exists() or summary_path.exists():
            raise ValueError('choose a new BC seed archive and summary path')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)/'state'
            fork_bc(source, state/'run', max_wall_time_seconds=max_wall_time_seconds, experimental=experimental)
            summary = verify(state)
            summary['sha256'] = seal(state, output)
            write(summary_path, summary)


def credentials(directory, run, expiry):
    runner, owner = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    cap = dict(endpoint=trusted_endpoint(), run=run)
    keydir = Path.home()/'.local/share/brass-kaggle-control'/run
    keydir.mkdir(mode=0o700, parents=True, exist_ok=False)
    write(keydir/'owner.json', dict(cap, token=owner))
    (keydir/'owner.json').chmod(0o600)
    write(directory/'runner.json', dict(cap, token=runner))
    write(directory/'registration.json', dict(run=run, expiry=expiry,
          runner_hash=hashlib.sha256(runner.encode()).hexdigest(), owner_hash=hashlib.sha256(owner.encode()).hexdigest()))


def campaign_gate(proof, state, summary_path, deadline, revision):
    from owner import download, provider_status
    from campaign import allow_long
    allow_long(proof)
    prior = read(proof/'job.json')
    if ((proof/'operator-halt.json').exists() or prior.get('stage') != 'proof' or deadline != prior['deadline'] or
            revision != prior['revision'] or prior['initial']['update'] != BASE or
            state.resolve() != (proof/'verified/continuation.tar.gz').resolve() or
            summary_path.resolve() != (proof/'verified/verified.json').resolve()):
        raise ValueError('wrong proof, input paths, campaign deadline or revision')
    if provider_status(prior['kernel']) != 'COMPLETE':
        raise ValueError('proof publication is incomplete')
    result = download(proof,prior)
    stop = read(proof/'stop-receipt.json')
    if (result['update'] <= BASE or result['update'] % 10 == 0 or result['next_game_id'] != 2*result['update'] or
            result['reason'] != 'manual' or result['outcome'] != 'normal' or not result['ack_at'] or
            result['requested_at'] != round(stop['requested_at']*1000) or
            result['kernel'] != prior['kernel'] or result['control_id'] != prior['run'] or
            stop['run'] != prior['run'] or stop['requested'] is not True or stop.get('purpose') != 'proof' or
            result['experiment'] != prior['initial']['experiment'] or not result['replay_count']):
        raise ValueError('proof requires a verified off-cadence manual stop after real training')
    return prior['run']


def cpu_wheel():
    path = Path.home()/'.cache/brass-kaggle'/WHEEL_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            candidate = Path(tmp)/WHEEL_NAME
            url = 'https://download-r2.pytorch.org/whl/cpu/'+WHEEL_NAME.replace('+','%2B')
            with urllib.request.urlopen(url,timeout=60) as stream, candidate.open('xb') as out:
                shutil.copyfileobj(stream,out)
            if digest(candidate) != WHEEL_SHA:
                raise ValueError('official CPU wheel digest mismatch')
            os.replace(candidate,path)
    if digest(path) != WHEEL_SHA:
        raise ValueError('cached CPU wheel digest mismatch')
    return path


def prepare(state, summary_path, dest, slug, deadline, stage='resume', proof=None, *, project):
    if not time.time()+2400 < deadline <= time.time()+12*3600+1:
        raise ValueError('deadline must provide export reserve within 12 hours')
    summary = read(summary_path)
    if digest(state) != summary['sha256']:
        raise ValueError('input bytes differ from verified checkpoint archive')
    frozen = read(proof/'job.json')['revision'] if stage == 'long' and proof is not None else 'HEAD'
    revision = subprocess.check_output(['git','rev-parse','--verify',frozen+'^{commit}'],cwd=ROOT,text=True).strip()
    previous = None
    if summary.get('learner') and stage != 'resume':
        raise ValueError('supported BC uses the resume stage and its own sealed source')
    if stage == 'proof' and (summary['update'] != BASE or summary['policy'] != POLICY or summary['replay_count']):
        raise ValueError('proof must start at the sealed 843 seed')
    if stage == 'long':
        if proof is None:
            raise ValueError('long campaign segment requires the verified proof directory')
        previous = campaign_gate(proof,state,summary_path,deadline,revision)
    session_deadline = min(deadline,int(time.time())+60*60) if stage == 'proof' else deadline
    dest.mkdir(mode=0o700, parents=True, exist_ok=False)
    data, kernel = dest/'dataset', dest/'kernel'
    data.mkdir()
    kernel.mkdir()
    run = secrets.token_hex(16)
    if run == previous:
        raise ValueError('new segment requires a fresh control ID')
    credentials(dest, run, int(deadline)+7*86400)
    shutil.copyfile(dest/'runner.json', data/'runner.json')
    shutil.copyfile(state, data/'state.bin')
    shutil.copyfile(cpu_wheel(),data/'torch-cpu.bin')
    paths = ['Cargo.lock','Cargo.toml','engine/core','engine/bindings','engine/tools','players/baselines',
             'training/brass_rl','training/requirements/linux.txt',
             'training/scripts/kaggle_control/common.py','training/scripts/kaggle_control/job.py',
             'training/scripts/kaggle_control/train_run.py']
    if summary.get('learner'):
        paths.append('training/configs')
    with (data/'source.bin').open('xb') as out:
        subprocess.run(['git','archive','--format=tar.gz',revision]+paths, stdout=out, cwd=ROOT, check=True)
    if stage == 'long' and digest(data/'source.bin') != read(proof/'dataset/run-input.json')['files']['source.bin']:
        raise ValueError('scientific payload differs from the verified proof revision')
    write(data/'run-input.json', dict(run=run, revision=revision, deadline=deadline, session_deadline=session_deadline,
          stage=stage, proof_run=previous, initial=summary,
          control_endpoint=trusted_endpoint(), files={name:digest(data/name) for name in ('state.bin','source.bin','runner.json','torch-cpu.bin')}))
    descriptor = digest(data/'run-input.json')
    launcher = subprocess.check_output(['git','show',revision+':training/scripts/kaggle_control/launch.py'],cwd=ROOT,text=True)
    (kernel/'run.py').write_text(launcher.replace('__DESCRIPTOR_SHA256__', descriptor))
    account, name = slug.split('/')
    dataset = account+'/'+name+'-input'
    title = name.replace('-',' ').title()
    write(data/'dataset-metadata.json', dict(id=dataset, title=title+' Input', licenses=[dict(name='other')]))
    write(kernel/'kernel-metadata.json', dict(id=slug, title=title, code_file='run.py', language='python',
          kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,
          dataset_sources=[dataset],kernel_sources=[],model_sources=[],competition_sources=[]))
    write(dest/'job.json', dict(library_project=str(Path(project).resolve()), run=run, kernel=slug+'/1', dataset=dataset+'/versions/1', deadline=deadline,
          descriptor=descriptor,revision=revision,stage=stage,proof_run=previous,
          proof_directory=proof.name if proof is not None else None,session_deadline=session_deadline,initial=summary))


def claim_successor(previous, directory, plan):
    from campaign import reserve_long
    from owner import owner_lock
    with owner_lock(previous.parent):
        if reserve_long(previous,directory,plan):
            return
        with owner_lock(previous):
            if (previous/'operator-halt.json').exists():
                raise ValueError('owner ended the proof campaign before long submission')
            successor = previous/'successor.json'
            if successor.exists():
                raise ValueError('this proof already has a submitted or reserved successor')
            write(successor,dict(run=plan['run'],directory=str(directory.resolve())))


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    cmd = sub.add_parser('seed')
    cmd.add_argument('project', type=Path)
    cmd.add_argument('archive', type=Path)
    cmd = sub.add_parser('seed-bc', help='activate a new supported BC run from an explicit full checkpoint')
    cmd.add_argument('source', type=Path)
    cmd.add_argument('archive', type=Path)
    cmd.add_argument('--max-wall-time-seconds', type=float, help='training wall budget for the new BC child')
    cmd.add_argument('--experimental', action='store_true')
    cmd = sub.add_parser('prepare')
    cmd.add_argument('--project-root', type=Path, required=True, help='host data project receiving returned models and games')
    for name in ('state','summary','directory'):
        cmd.add_argument(name, type=Path)
    cmd.add_argument('slug')
    cmd.add_argument('deadline', type=int)
    cmd.add_argument('--stage', choices=['proof','long','resume'],default='resume')
    cmd.add_argument('--proof',type=Path)
    cmd = sub.add_parser('adopt-retry')
    cmd.add_argument('original',type=Path)
    cmd.add_argument('retry',type=Path)
    cmd = sub.add_parser('submit')
    cmd.add_argument('directory',type=Path)
    args = parser.parse_args()
    if args.cmd == 'seed':
        seed(args.project.resolve(), args.archive.resolve())
    elif args.cmd == 'seed-bc':
        seed_bc(args.source.resolve(), args.archive.resolve(), max_wall_time_seconds=args.max_wall_time_seconds, experimental=args.experimental)
    elif args.cmd == 'adopt-retry':
        from campaign import adopt_retry
        state = adopt_retry(args.original,args.retry)
        if state.get('halted_by'):
            subprocess.run([sys.executable,HERE/'owner.py',args.original,'stop'],check=True)
    elif args.cmd == 'submit':
        from owner import kaggle
        directory = args.directory.resolve()
        plan = read(directory/'job.json')
        now = time.time()
        remaining = int(min(plan['deadline'],plan['session_deadline'])-now)
        if remaining <= 2400:
            raise ValueError('insufficient campaign time for update allowance and export')
        if digest(directory/'dataset/run-input.json') != plan['descriptor']:
            raise ValueError('prepared descriptor changed')
        request = directory/'submission-attempt.json'
        # A failed/ambiguous submit needs explicit reconciliation with the provider.
        with request.open('x') as out:
            import json
            json.dump(dict(at=now,timeout=remaining,kernel=plan['kernel']),out)
            out.flush()
            os.fsync(out.fileno())
        if plan['stage'] == 'long':
            previous = directory.parent / Path(plan['proof_directory']).name
            if read(previous/'job.json')['run'] != plan['proof_run']:
                raise ValueError('local proof identity mismatch')
            claim_successor(previous,directory,plan)
        kaggle('kernels','push','-p',directory/'kernel','--timeout',remaining)
    else:
        prepare(args.state,args.summary,args.directory,args.slug,args.deadline,args.stage,args.proof,project=args.project_root)
