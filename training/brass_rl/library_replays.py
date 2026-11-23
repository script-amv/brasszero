# SPDX-License-Identifier: Apache-2.0
"""Reconcile full legacy exemplars with the action archive using bounded memory."""
from collections import defaultdict
from datetime import datetime, timezone
import json

from brass_ai.replay_archive import (Archive, FORMAT, canonical_game, digest, engine_id,
                                    legacy_source_metadata, outcome, seat_identities)
from .experiments import writer_lock


def _legacy_files(runs):
    files = []
    for run in runs:
        for path in sorted(set((run/'replays').glob('*.json')) |
                           set((run/'evaluation').glob('**/replays/*.json'))):
            if path.name == 'index.json':
                continue
            payload = json.loads(path.read_bytes())
            if payload.get('format') == 'brass-gui-game/1':
                c = canonical_game(payload)
                files.append((path, run, digest(c['initial']), _record_signature(c)))
    return files


def _record_signature(game):
    # Old GUI exemplars often lack the versioned seat fields of the durable recorder.
    return digest([game['initial'], game['moves'], [s.get('name') for s in game['seats']]])


def _identifier(value):
    return isinstance(value, str) and bool(value)


def _game_number(value):
    return type(value) is int or _identifier(value)


def _evaluation_key(suite, candidate, number):
    if _identifier(suite) and _identifier(candidate) and _game_number(number):
        return suite, candidate, str(number)
    return None


def _coverage(owner, number, seed, result):
    return (owner, str(number), seed, tuple(result.get('vp', [])), tuple(result.get('ranking', [])))


def _initial_id(initial):
    if isinstance(initial.get('start_state'), str):
        return digest({'start_state': json.loads(initial['start_state'])})
    return digest(initial)


def _evaluation_evidence(initial, seats, result):
    identities = seat_identities(seats, legacy=True)
    return (_initial_id(initial), result.get('vp'), result.get('ranking'),
            [(s.get('name'), ident.get('version_id')) for s, ident in zip(seats, identities)])


def _matches_outcome(row, evidence):
    initial, vp, ranking, seats = evidence
    if 'deal_seed' in row:
        expected = {'seed': row['deal_seed']}
    elif 'start_state' in row:
        expected = {'start_state': row['start_state']}
    else:
        return False
    if _initial_id(expected) != initial or vp != row.get('vp') or ranking != row.get('ranking'):
        return False

    def matches_seat(label, seat):
        name, identity = seat
        if not isinstance(label, str) or not label:
            return False
        if identity and label in ('random', 'greedy'):
            return identity == label or identity.startswith(label+':native-v1:')
        if identity and (label.startswith(('policy-sha256-v1:', 'greedy:native-', 'random:native-'))
                         or ':weights-sha256:' in label):
            return identity == label
        return label in (name, identity)

    labels = row.get('labels')
    if labels is not None and (len(labels) != len(seats) or
                               not all(matches_seat(label, seat) for label, seat in zip(labels, seats))):
        return False
    candidate = row.get('candidate_seat')
    if candidate is not None:
        return (type(candidate) is int and 0 <= candidate < len(seats)
                and matches_seat(row.get('candidate_id'), seats[candidate]))
    actor = row.get('candidate_id')
    return actor is None or any(matches_seat(actor, seat) for seat in seats)


def reconcile_replays(project, runs, journals):
    import brass
    from .transfer import file_hash
    archive = Archive(project/'replays')
    files = _legacy_files(runs)
    seeds = {seed for _, _, seed, _ in files}
    signatures = {sig for _, _, _, sig in files}
    matching = defaultdict(list)
    games, available = set(), set()
    evaluations = defaultdict(list)
    paths = {r.as_posix(): e['id'] for r, e, _, _ in journals}
    for name in sorted({p.name.split('.')[0] for p in archive.games.glob('*.json*') if not p.name.startswith('.')}):
        r = archive.read(name)
        if r['status'] != 'completed':
            continue
        p = r['provenance']
        available.add(name)
        who = p.get('run_experiment')
        if who is None:
            who = p.get('experiment')
        if r['category'] == 'training' and _identifier(who) and _game_number(p.get('game')):
            games.add(_coverage(paths.get(who, who), p['game'], r['initial'].get('seed'), r['result']))
        elif r['category'] == 'evaluation':
            key = _evaluation_key(p.get('suite_id', p.get('experiment')), p.get('candidate_id'),
                                  p.get('game_id', p.get('game')))
            if key is not None:
                evaluations[key].append(_evaluation_evidence(r['initial'], r['seats'], r['result']))
        if digest(r['initial']) in seeds:
            signature = _record_signature(r)
            if signature in signatures:
                matching[signature].append(name)
    count = 0
    incomplete = []
    for path, run, _, signature in files:
        c = canonical_game(json.loads(path.read_bytes()))
        sha = file_hash(path)
        info = legacy_source_metadata(path)
        who = paths.get(str(run))
        game_id = str(info.get('game_id', info.get('game')))
        # Bind the weakly labelled exemplar only to exact recorded moves, seats and origin.
        candidates = matching.get(signature, [])
        matched = False
        for name in candidates:
            record = archive.read(name)
            provenance = record['provenance']
            if (canonical_game(record) == c or any(s.get('sha256') == sha for s in provenance.get('sources', []))
                    or (who and who == provenance.get('experiment')
                    and str(provenance.get('game')) == game_id)):
                matched = True
                break
        if not matched or info['category'] == 'evaluation':
            game = brass.Game(c['initial']['seed']) if 'seed' in c['initial'] else brass.Game.from_json(c['initial']['start_state'])
            for ply in c['moves']:
                if game.is_over() or ply['player'] != game.current_player():
                    raise ValueError('invalid legacy replay actor: '+str(path))
                game.apply(json.dumps(ply['move']))
            if not game.is_over():
                incomplete.append(str(path.relative_to(project)))
                continue
            result = outcome(game)
            if matched and record['result'] != result:
                raise ValueError('archive outcome disagrees with native exemplar: '+str(path))
        else:
            result = record['result']
        if info['category'] == 'evaluation':
            row = json.loads((path.parent.parent/'games'/path.name).read_bytes())
            evidence = _evaluation_evidence(c['initial'], c['seats'], result)
            key = _evaluation_key(info.get('suite_id'), info.get('candidate_id'), info.get('game_id', info.get('game')))
            if key is not None and _matches_outcome(row, evidence):
                evaluations[key].append(evidence)
            else:
                info = dict(category='legacy', game=path.stem)
        if not matched:
            name = digest(c)
            p = dict(info, imported=True, sources=[dict(path=str(path), sha256=sha, provenance=info)])
            record = dict(c, format=FORMAT, game_id=name, engine_id=engine_id(), provenance=p,
                          category=info['category'], created_at=datetime.now(timezone.utc).isoformat(),
                          status='completed', result=result, seat_identities=seat_identities(c['seats'], legacy=True))
            with writer_lock(archive.root/'locks'/('transfer-'+name)):
                count += int(archive.publish(record))
            matching[signature].append(name)
            available.add(name)
        if info['category'] == 'training' and _identifier(who):
            games.add(_coverage(who, game_id, c['initial'].get('seed'), record['result']))
    missing_eval = set()
    for run in runs:
        for path in (run/'evaluation').glob('**/games/*.json'):
            game = json.loads(path.read_bytes())
            key = _evaluation_key(game.get('suite_id'), game.get('candidate_id'), game.get('game_id'))
            if not any(_matches_outcome(game, evidence) for evidence in evaluations.get(key, ())):
                missing_eval.add(str(path.relative_to(project)))
    missing, missing_ids = set(), set()
    for run, experiment, update, journal in journals:
        owner = journal.get('experiment_id', experiment['id'])
        if not _identifier(owner):
            owner = None
        aliases = [owner] if owner else []
        legacy = experiment.get('legacy_replay_identity')
        if owner == experiment['id'] and _identifier(legacy):
            aliases.append(legacy)
        for row in journal.get('train', []):
            if not any(_coverage(alias, row['game_id'], row.get('deal_seed'), row.get('outcome', {})) in games for alias in aliases):
                missing.add((owner, update, row['game_id']))
        missing_ids.update(set(journal.get('replay_ids', []))-available)
    indexed = len(archive.rebuild())
    gaps = sorted(missing, key=lambda item: (item[0] or '', item[1], item[2]))
    return dict(indexed_replays=indexed, legacy_files=len(files), added_replays=count,
                missing_training_games=len(missing), missing_evaluation_games=len(missing_eval),
                missing_evaluation_rows=sorted(missing_eval),
                missing_games=[dict(experiment_id=e, update=n, game=g) for e,n,g in gaps],
                missing_linked_replays=sorted(missing_ids), incomplete_legacy_files=incomplete)
