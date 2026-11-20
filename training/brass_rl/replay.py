"""Bounded passive GUI recordings; never import the GUI server or bot registry."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import brass
from brass_ai.events import PublicEvent
from brass_ai.replay_archive import digest


def experiment_identity(config, master):
    """Stable across exactly the operational overrides allowed by resume."""
    learning = asdict(config)
    # v1 identities predate optional study settings. Omit their neutral values
    # so replay identities of restored legacy configs remain byte-compatible.
    for section, key, default in (('ppo', 'reward', 'strict_win'), ('ppo', 'critic_target', 'gae'),
                                  ('league', 'lineup_schedule', 'sample')):
        if learning[section][key] == default:
            del learning[section][key]
    for name in ('device', 'max_wall_time_seconds', 'checkpoint_margin_seconds'):
        learning['runtime'].pop(name)
    return digest({'master': master, 'config': learning})


def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(payload, output, sort_keys=True, separators=(',', ':'), allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_gui_replay(path: Path, seed: int, labels: Sequence[str], events: Sequence[PublicEvent],
                     *, expected_ranking=None, expected_vp=None) -> None:
    """Reconstruct a complete game before atomically publishing its seed/moves.

    Public snapshots are checked but not stored. Seed stays a JSON integer (u64),
    including values beyond Javascript's exact Number range.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('replay seed must be an unsigned 64-bit integer')
    if len(labels) != 4 or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError('replay needs four nonempty identity labels')
    game, moves = brass.Game(seed), []
    for ply, event in enumerate(events, 1):
        if not isinstance(event, PublicEvent) or event.ply != ply or game.is_over() or event.actor != game.current_player():
            raise ValueError('replay events must have consecutive plies and correct actors')
        move = json.loads(event.move_json)
        legal = [json.loads(m) for m in game.observation(event.actor).legal_moves()]
        try:
            index = legal.index(move)
        except ValueError as exc:
            raise ValueError('replay move is not legal') from exc
        game.apply_index(index)
        if json.loads(game.observation(event.actor).public_json()) != json.loads(event.public_json):
            raise ValueError('replay public outcome mismatch')
        moves.append({'player': event.actor, 'move': move})
    if not game.is_over():
        raise ValueError('replay must contain a complete game')
    if ((expected_ranking is not None and tuple(game.ranking()) != tuple(expected_ranking)) or
            (expected_vp is not None and tuple(game.vp()) != tuple(expected_vp))):
        raise ValueError('reconstructed replay ranks/VP mismatch recorded result')
    atomic_json(path, {'format': 'brass-gui-game/1', 'seed': seed,
                      'seats': [{'kind': 'human', 'name': label, 'bot': None} for label in labels], 'moves': moves})


def _outcome(row):
    seat = row.get('candidate_seat', 0)
    vp = row['vp'][seat]
    rank = row['ranking'][seat] + 1
    margin = vp - max(value for i, value in enumerate(row['vp']) if i != seat)
    return vp, rank, margin


def select_replays(records) -> Mapping[str, int]:
    """Stable exemplar IDs; the median is the lower median for even populations."""
    rows = list(records)
    if not rows:
        return {}
    ordered = sorted(rows, key=lambda r: (_outcome(r)[0], r['game_id']))
    median_vp = _outcome(ordered[(len(rows)-1)//2])[0]
    median = min((r for r in rows if _outcome(r)[0] == median_vp), key=lambda r: r['game_id'])
    return {'best_vp': min(rows, key=lambda r: (-_outcome(r)[0], r['game_id']))['game_id'],
            'lowest_vp': ordered[0]['game_id'],
            'worst_rank': min(rows, key=lambda r: (-_outcome(r)[1], _outcome(r)[2], r['game_id']))['game_id'],
            'median_vp': median['game_id']}


def retained_ids(records, limit):
    """Exemplars followed by a deterministic bounded sample, with stable ties."""
    if type(limit) is not int or not 0 <= limit <= 100:
        raise ValueError('replay limit must be in 0..100')
    rows = list(records)
    exemplars = list(dict.fromkeys(select_replays(rows).values()))
    return (exemplars + [r['game_id'] for r in sorted(rows, key=lambda r: r['game_id'])
                         if r['game_id'] not in exemplars])[:limit]


def retain_training_replays(directory: Path, episodes, streams, limit: int, candidate_id: str,
                            frozen_identities: Mapping[str, str]):
    """Keep <=limit complete episode histories across updates, not all histories.

    The index/replays are disposable after-commit inspection files. A resume never
    replays training games to repair them; damaged/missing index files can be removed.
    Selection is over retained recordings plus this complete batch (bounded memory).
    Frozen labels come from controller-verified collection snapshots, not run-local aliases.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = directory/'index.json'
    old = json.loads(index.read_text()) if index.exists() else []
    old = [r for r in old if (directory/('%s.json' % r['game_id'])).is_file()]
    episodes = {e.game_id: e for e in episodes}
    records = {r['game_id']: r for r in old}
    for episode in episodes.values():
        labels = []
        for entry in episode.lineup.entries:
            if entry.kind == 'current':
                labels.append(candidate_id)
            elif entry.kind == 'frozen':
                if entry.snapshot_id not in frozen_identities:
                    raise ValueError('frozen replay identity missing for snapshot '+entry.snapshot_id)
                labels.append(frozen_identities[entry.snapshot_id])
            else:
                labels.append(entry.kind)
        records[episode.game_id] = {'game_id': episode.game_id, 'candidate_seat': next(
            i for i, e in enumerate(episode.lineup.entries) if e.trainable),
            'vp': list(episode.result.vp), 'ranking': list(episode.result.ranking), 'labels': labels}
    keep = retained_ids(records.values(), limit)
    # Remove unretained histories before writing; cap is respected across updates.
    for path in directory.glob('*.json'):
        if path.name != 'index.json' and path.stem not in {str(i) for i in keep}:
            path.unlink()
    for game_id in keep:
        if game_id in episodes:
            ep, row = episodes[game_id], records[game_id]
            write_gui_replay(directory/('%s.json' % game_id), streams.seed('train.deal', game_id), row['labels'],
                             ep.events, expected_ranking=ep.result.ranking, expected_vp=ep.result.vp)
    atomic_json(index, [records[i] for i in keep])
    return tuple(keep)
