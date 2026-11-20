# SPDX-License-Identifier: Apache-2.0
"""Bind the first dynamic collection to an immutable legacy source boundary."""
import hashlib
import re

from .dynamic_league import DynamicLeague
from .experiments import atomic, encoded
from .league import Snapshot, replay_identity
from .policy_io import read_json
from .retention import file_sha


def _source(boundary, state, experiment):
    return dict(format='brass-dynamic-activation/1', experiment=experiment,
                source_manifest_sha256=file_sha(boundary/'manifest.json'),
                update=state.completed_updates, next_game_id=state.next_game_id)


def activation_league(run, boundary, state, streams, experiment, requested):
    path = run/'dynamic-activation.json'
    if path.exists():
        seal = read_json(path)
        rows = seal.pop('members')
        if seal.pop('members_sha256') != hashlib.sha256(encoded(rows)).hexdigest():
            raise ValueError('sealed dynamic membership differs from its recorded digest')
        if seal != _source(boundary, state, experiment):
            raise ValueError('dynamic activation belongs to a different experiment or source boundary')
        members = []
        for row in rows:
            digest = row['sha256']
            if not re.fullmatch(r'[0-9a-f]{64}',digest):
                raise ValueError('invalid sealed dynamic member digest')
            weights = run/'weights'/(digest+'.pt')
            if file_sha(weights)!=digest:
                raise ValueError('sealed dynamic member bytes differ')
            members.append(Snapshot(row['id'], weights, row['update'], row['role']))
    else:
        members = requested(state.completed_updates) if callable(requested) else requested
    return DynamicLeague.start(streams, members, state.holdouts, history_extent=state.completed_updates+1,
                               next_game_id=state.next_game_id)


def seal_activation(run, boundary, effective, model, experiment):
    from .artifacts import store_snapshot_records
    rows = store_snapshot_records(run,effective,model)
    members = [row for row in rows if row['role']!='holdout']
    if any(row['sha256']!=effective.league_history[row['id']]['sha256'] for row in rows):
        raise ValueError('dynamic member changed before activation')
    seal = dict(_source(boundary,effective,experiment),members=members,
                members_sha256=hashlib.sha256(encoded(members)).hexdigest())
    atomic(run/'dynamic-activation.json',encoded(seal))


def check_pending_training(league, state, experiment, candidate, first_activation):
    """Check prior recording identities before an interrupted update can write."""
    from brass_ai.replay_archive import Archive, digest, engine_id, labeled_seats
    identity = experiment.get('legacy_replay_identity',experiment['id'])
    update = state.completed_updates+1
    games = range(state.next_game_id,state.next_game_id+state.config.games_per_update)
    history = league.history
    for record in Archive().records(include_duplicates=True):
        origin = record['provenance']
        if record['category']!='training' or origin.get('experiment')!=identity:
            continue
        game = origin.get('game')
        recorded_update = origin.get('update')
        if type(game) is not int or type(recorded_update) is not int:
            raise ValueError('incomplete training recording provenance for this experiment')
        if recorded_update<update and game<games.start:
            continue
        if first_activation:
            raise ValueError('pending training recording at legacy boundary; resume its current mode first')
        if game not in games or origin.get('update')!=update:
            raise ValueError('pending training recording lies outside the next collection')
        lineup = league.sample(game)
        labels = [candidate if e.trainable else replay_identity(e.snapshot_id,history[e.snapshot_id])
                  if e.kind=='frozen' else e.kind for e in lineup.entries]
        expected = dict(initial={'seed':league.streams.seed('train.deal',game)},seats=labeled_seats(labels),
                        provenance=dict(category='training',experiment=identity,update=update,game=game),
                        engine_id=engine_id())
        actual = {key:record[key] for key in expected}
        actual['provenance'] = {k:v for k,v in origin.items() if k!='experiment_path'}
        if actual!=expected or record['game_id']!=digest(expected) or record['status'] not in ('completed','interrupted'):
            raise ValueError('pending training recording conflicts with the restored league or game identity')
