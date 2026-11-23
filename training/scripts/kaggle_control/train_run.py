# SPDX-License-Identifier: Apache-2.0
"""CPU numerical child. File observation occurs only at the trainer's update boundary."""
import argparse
from pathlib import Path

from common import BASE, POLICY, SOURCE, digest, read, write


def _bc_source(run, exp, restored):
    from brass_rl.artifacts import load_update
    from brass_rl.config import BC_LEARNER, TrainConfig

    parent = exp['parent']
    base = parent['update']
    initial = load_update(run/'updates'/('%06d' % base))
    first = initial.state.next_game_id
    checks = exp['initial_state_verified']
    critic_checks = (('critic_clone', 'fresh_critic_adam') if exp['critic_initialization'] == 'cloned'
                     else ('critic_restored', 'critic_adam_restored'))
    required = {'actor', 'actor_adam', 'global_rng', 'shuffle_rng', 'league', 'counters_and_master_seed',
                'public_policy_bytes', 'loss_priority', *critic_checks}
    if (exp.get('mode') != 'bc-current' or exp.get('learner') != BC_LEARNER
            or exp['critic_initialization'] not in ('cloned', 'restored')
            or exp['loss_priority_initialization'] not in ('empty', 'restored')
            or exp['base_update'] != base or exp['initial_next_game_id'] != first
            or checks.keys() != required or not all(value is True for value in checks.values())
            or read(initial.path/'metadata.json')['id'] != parent['model_id']
            or digest(initial.path/'policy.pt') != parent['policy_sha256']
            or initial.state.manifests['source_manifest_sha256'] != parent['checkpoint_sha256']
            or initial.state.config.learner != BC_LEARNER or initial.state.config != restored.state.config
            or TrainConfig.from_json(run/'config.json') != initial.state.config):
        raise ValueError('supported BC source or fork declaration differs from checkpoint')
    if (exp['loss_priority_initialization'] == 'empty' and
            (initial.state.loss_priority['counts'] or initial.state.loss_priority['first_game_id'] != first)):
        raise ValueError('invalid initial loss-priority exposure')
    return base, first


def _check_bc_batch(league, current, batch, by_game, expected_exposure):
    from dataclasses import asdict
    from brass_ai.replay_archive import labeled_seats
    from brass_rl.league import replay_identity

    # All eight draws use the previous boundary's counts, including in mixed games.
    lineups = [league.sample(g['game_id']) for g in batch]
    for game, lineup in zip(batch, lineups):
        labels = [current if e.trainable else replay_identity(e.snapshot_id, league.history[e.snapshot_id])
                  if e.kind == 'frozen' else e.kind for e in lineup.entries]
        if ([asdict(e) for e in lineup.entries] != game['lineup']
                or by_game[game['game_id']]['seats'] != labeled_seats(labels)
                or game['deal_seed'] != league.streams.seed('train.deal', game['game_id'])):
            raise ValueError('supported BC draw, deal or frozen seat identity differs from archive')
        league.record_training(game['game_id'], lineup, game['outcome']['ranking'])
    if league.state != expected_exposure:
        raise ValueError('committed loss-priority counts differ from native training ranks')


def verify(root, native=True):
    from brass_ai.replay_archive import Archive
    from brass_rl.artifacts import committed_records, inspect_update_state, load_update, restore_league
    from brass_rl.league_study import check_legacy_treatment
    from brass_rl.policy_io import inspect_policy
    from brass_rl.retention import present_payloads
    from brass_rl.transfer import validate_replay
    run = root/'run'
    exp = read(run/'experiment.json')
    rows = committed_records(run)
    tip = rows[-1]['update']
    restored = load_update(run/'latest')
    s = restored.state
    opt_steps = [int(v['step']) for v in restored.optimizer_state['state'].values()]
    if s.config.learner:
        base, first = _bc_source(run, exp, restored)
        critic_steps = [int(v['step']) for v in restored.critic_optimizer_state['state'].values()]
        extra = dict(learner=s.config.learner, critic_adam_slots=len(critic_steps),
                     critic_adam_step=min(critic_steps, default=0), loss_priority_next_game_id=s.loss_priority['next_game_id'],
                     base_update=base, initial_next_game_id=first, source_manifest=exp['parent']['checkpoint_sha256'])
    else:
        base, first = BASE, 2*BASE
        parent = exp['parent']
        if (exp.get('learner') or parent['checkpoint_sha256'] != SOURCE or parent['model_id'] != POLICY
                or exp['config_overrides'] or not all(exp['initial_state_verified'].values())):
            raise ValueError('legacy continuation parent or initial state differs')
        if (len(opt_steps) != 53 or len(set(opt_steps)) != 1 or min(opt_steps) < 18729
                or len(s.league_snapshots) != 8 or len(s.holdouts) != 1 or s.config.games_per_update != 2):
            raise ValueError('invalid legacy optimizer or league')
        extra = {}
    batch_size = s.config.games_per_update
    if (s.completed_updates != tip or s.next_game_id != first+batch_size*(tip-base) or s.dynamic_league is not None
            or s.config.eval.interval_updates or s.config.runtime.device != 'cpu' or s.config.runtime.torch_threads != 1):
        raise ValueError('invalid cloud tip/config')
    if [r['update'] for r in rows] != list(range(base, tip+1)):
        raise ValueError('journal gap')
    records = Archive(root/'replays').records(include_duplicates=True)
    by_game = {r['provenance']['game']:r for r in records}
    if (len(records) != s.next_game_id-first or set(by_game) != set(range(first, s.next_game_id)) or
            any(r['status'] != 'completed' or r['category'] != 'training' or
                r['provenance']['experiment'] != exp['id'] for r in records)):
        raise ValueError('replay coverage mismatch')
    games = []
    league = current = None
    for row in rows:
        n = row['update']
        path = run/'updates'/('%06d'%n)
        manifest = read(path/'manifest.json')
        for name in present_payloads(path, manifest):
            if digest(path/name) != manifest['content_hashes'][name]:
                raise ValueError('bad checkpoint digest')
        if n in (base,tip) or n%10 == 0:
            if not (path/'optimizer.pt').is_file():
                raise ValueError('required full optimizer unavailable')
        public = run/'policies'/path.name
        meta = inspect_policy(public)
        if meta.id != read(path/'metadata.json')['id'] or (n == base and meta.id != exp['parent']['model_id']):
            raise ValueError('public export identity mismatch')
        m = row['manifests']
        batch = m.get('train', [])
        start = first + batch_size*(n-base-1)
        expected = [] if n == base else list(range(start, start+batch_size))
        if ([g['game_id'] for g in batch] != expected or m.get('dev') or m.get('final')
                or len(m['replay_ids']) != len(expected)
                or set(m['replay_ids']) != {by_game[i]['game_id'] for i in expected}):
            raise ValueError('replay or update identity mismatch')
        for g in batch:
            r = by_game[g['game_id']]
            if (r['initial'] != {'seed':g['deal_seed']} or r['provenance']['update'] != n or
                    any(r['result'][k] != g['outcome'][k] for k in ('vp','ranking'))):
                raise ValueError('replay disagrees with native game journal')
        if s.config.learner and native:
            saved = inspect_update_state(path, s.config)
            if n > base:
                _check_bc_batch(league, current, batch, by_game, saved.loss_priority)
            league, current = restore_league(saved), meta.id
        elif not s.config.learner:
            games.extend(batch)
    if native:
        for record in records:
            validate_replay(record)
        if not s.config.learner:
            check_legacy_treatment(run, games, by_game)
    return dict(extra, update=tip, experiment=exp['id'], policy=read(restored.path/'metadata.json')['id'],
                next_game_id=s.next_game_id, replay_count=len(records), adam_step=min(opt_steps, default=0),
                adam_slots=len(opt_steps), manifest=digest(restored.path/'manifest.json'))


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
        report = advance(TrainConfig.from_json(run/'config.json'), run, run/'latest',
                         max_updates=999999, stop_requested=boundary)
    notice = signals/'child-stop.json'
    if report.stop_reason == 'wall_time' and not notice.exists():
        write(notice, dict(reason='deadline'))


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
