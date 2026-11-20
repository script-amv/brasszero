# SPDX-License-Identifier: Apache-2.0
"""CPU numerical child. File observation occurs only at the trainer's update boundary."""
import argparse
from pathlib import Path

from common import BASE, POLICY, SOURCE, digest, read, write


def verify(root, native=True):
    from brass_ai.replay_archive import Archive
    from brass_rl.artifacts import committed_records, load_update
    from brass_rl.league_study import check_legacy_treatment
    from brass_rl.policy_io import inspect_policy
    from brass_rl.retention import present_payloads
    from brass_rl.transfer import validate_replay
    run = root/'run'
    exp = read(run/'experiment.json')
    parent = exp['parent']
    if (parent['checkpoint_sha256'] != SOURCE or parent['model_id'] != POLICY or
            exp['config_overrides'] or not all(exp['initial_state_verified'].values())):
        raise ValueError('continuation parent or initial state differs')
    rows = committed_records(run)
    tip = rows[-1]['update']
    restored = load_update(run/'latest')
    s = restored.state
    if (s.completed_updates != tip or s.next_game_id != 2*tip or s.dynamic_league is not None or
            s.config.eval.interval_updates or s.config.runtime.device != 'cpu' or
            s.config.runtime.torch_threads != 1 or len(s.league_snapshots) != 8 or len(s.holdouts) != 1):
        raise ValueError('invalid legacy tip/config')
    opt_steps = {int(v['step']) for v in restored.optimizer_state['state'].values()}
    if len(restored.optimizer_state['state']) != 53 or len(opt_steps) != 1 or min(opt_steps) < 18729:
        raise ValueError('unexpected Adam state')
    if [r['update'] for r in rows] != list(range(BASE, tip+1)):
        raise ValueError('journal gap')
    records = Archive(root/'replays').records(include_duplicates=True)
    by_game = {r['provenance']['game']:r for r in records}
    if (len(records) != 2*(tip-BASE) or set(by_game) != set(range(2*BASE, 2*tip)) or
            any(r['status'] != 'completed' or r['category'] != 'training' or
                r['provenance']['experiment'] != exp['id'] for r in records)):
        raise ValueError('replay coverage mismatch')
    games = []
    for row in rows:
        n = row['update']
        path = run/'updates'/('%06d'%n)
        manifest = read(path/'manifest.json')
        for name in present_payloads(path, manifest):
            if digest(path/name) != manifest['content_hashes'][name]:
                raise ValueError('bad checkpoint digest')
        if n in (BASE,tip) or n%10 == 0:
            if not (path/'optimizer.pt').is_file():
                raise ValueError('required full optimizer unavailable')
        public = run/'policies'/path.name
        meta = inspect_policy(public)
        if meta.id != read(path/'metadata.json')['id'] or (n == BASE and meta.id != POLICY):
            raise ValueError('public export identity mismatch')
        m = row['manifests']
        batch = m.get('train', [])
        expected = [] if n == BASE else [2*n-2, 2*n-1]
        if ([g['game_id'] for g in batch] != expected or m.get('dev') or m.get('final') or
                set(m['replay_ids']) != {by_game[i]['game_id'] for i in expected}):
            raise ValueError('replay or update identity mismatch')
        for g in batch:
            r = by_game[g['game_id']]
            if (r['initial'] != {'seed':g['deal_seed']} or r['provenance']['update'] != n or
                    any(r['result'][k] != g['outcome'][k] for k in ('vp','ranking'))):
                raise ValueError('replay disagrees with native game journal')
        games.extend(batch)
    if native:
        for record in records:
            validate_replay(record)
        check_legacy_treatment(run, games, by_game)
    return dict(update=tip, experiment=exp['id'], policy=read(restored.path/'metadata.json')['id'],
                next_game_id=s.next_game_id, replay_count=len(records), adam_step=min(opt_steps),
                adam_slots=53, manifest=digest(restored.path/'manifest.json'))


def time_for_update(now, cutoff, previous_seconds):
    # The controller cutoff already includes ten minutes of first-update headroom.
    extra = max(0,2*previous_seconds-600)
    return now + extra < cutoff


def train(root, signals):
    import os
    import time
    import torch
    from brass_rl.__main__ import _graceful_signals
    from brass_rl.train import StopFlag, train as advance
    from brass_rl.config import TrainConfig
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stop = StopFlag()
    admitted = None
    cutoff = float(os.environ['BRASS_TRAIN_CUTOFF'])
    def boundary():
        nonlocal admitted
        now = time.monotonic()
        duration = 0 if admitted is None else now-admitted
        if (signals/'stop').exists():
            stop.request()
        elif not time_for_update(now,cutoff,duration):
            write(signals/'child-stop.json',dict(reason='deadline'))
            stop.request()
        heart = signals/'heartbeat'
        if heart.exists() and time.time()-heart.stat().st_mtime > 180:
            if not (signals/'child-stop.json').exists():
                write(signals/'child-stop.json',dict(reason='control_unavailable'))
            stop.request()
        if not stop():
            admitted = now
            (signals/'active').touch()
        return stop()
    run = root/'run'
    with _graceful_signals(stop):
        advance(TrainConfig.from_json(run/'config.json'), run, run/'latest',
                max_updates=999999, stop_requested=boundary)


def recover(root, control_id):
    from brass_ai.replay_archive import Archive
    from brass_rl.artifacts import committed_records
    from brass_rl.config import TrainConfig
    from brass_rl.train import train as advance
    run = root/'run'
    rows = committed_records(run)
    committed = {rid for row in rows for rid in row['manifests'].get('replay_ids',())}
    pending = root/'recovery-evidence'/control_id
    for recording in (root/'replays/games').glob('*'):
        if recording.is_file() and recording.name.split('.')[0] not in committed:
            pending.mkdir(parents=True,exist_ok=True)
            recording.rename(pending/recording.name)
    Archive(root/'replays').rebuild()
    tip = rows[-1]['update']
    advance(TrainConfig.from_json(run/'config.json'),run,run/'latest',max_updates=tip)
    from brass_rl.experiments import atomic, encoded
    atomic(run/'latest',encoded(dict(update='%06d'%tip)))
    verify(root,native=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train','verify','recover'])
    parser.add_argument('state', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.state, args.output)
    elif args.command == 'recover':
        recover(args.state,str(args.output))
    else:
        write(args.output, verify(args.state))
