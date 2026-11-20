# SPDX-License-Identifier: Apache-2.0
"""Single-writer complete-update controller; only committed journal rows survive.

Collection freezes the policy/lineups for complete games. Targets and PPO live in
update_policy. Graceful requests finish that entire update; exceptions leave the
preceding boundary resumable. No partial rollout or true game state is persisted.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import tempfile
import time
from typing import Callable, Optional, Tuple
import warnings

import numpy as np
import torch

from .retention import maintain_rolling, ROLLING_POLICY
from .artifacts import (RunState, capture_rng_states, committed_records, export_policy, load_update,
                        restore_rng_states, save_update, policy_id, restore_league)
from .config import TrainConfig
from .evaluate import periodic_development
from .replay import retain_training_replays, experiment_identity  # noqa: F401
from .dynamic_league import DynamicLeague
from .league import League, Snapshot, replay_identity
from .model import BrassPolicy
from .policy_io import json_bytes, tensor_bytes, read_json
from .experiments import ensure_experiment, preflight, project_for, writer_lock
from .ppo import update_policy
from .returns import strict_outcomes
from .rollout import collect_batch
from brass_ai.replay_archive import recording_context
from .runtime import runtime_check
from .schema import ComponentData
from .seeding import SeedStreams


@dataclass
class StopFlag:
    """Signal-independent callable; safe signal handlers only set this flag."""
    requested: bool = False

    def request(self):
        self.requested = True

    def __call__(self):
        return self.requested


@dataclass(frozen=True)
class TrainReport:
    latest_checkpoint: Path
    latest_policy: Path
    completed_updates: int
    next_game_id: int
    model: BrassPolicy
    state: RunState
    # Ordered (game ID, SHA256 of every recorded current-seat choice) pairs.
    choice_trace: Tuple[Tuple[int, str], ...]
    stop_reason: str
    rng_restore_notes: Tuple[str, ...] = ()
    preflight: Optional[dict] = None


def rebuild_metrics(run_dir: Path):
    """Atomically rebuild the disposable human log from committed updates only."""
    run_dir = Path(run_dir)
    journal = committed_records(run_dir)
    fd, temporary = tempfile.mkstemp(prefix='.metrics-', dir=str(run_dir))
    try:
        with os.fdopen(fd, 'wb') as output:
            for update in journal:
                for row in update['metrics']:
                    output.write(json_bytes(row))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, run_dir/'metrics.jsonl')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return journal


def publish_policies(run_dir: Path, journal=None) -> Path:
    """Publish/repair public three-file exports after trusted journal commit.

    Existing IDs are immutable: export_policy validates/reuses the selected latest
    directory. Missing older exports are repaired on resume without re-running
    collection/PPO. Failure cannot roll back an already committed journal entry.
    """
    run_dir = Path(run_dir)
    journal = committed_records(run_dir) if journal is None else journal
    latest = None
    project = project_for(run_dir)
    published = {} if project is None else {read_json(p)['id']: p.parent
                 for p in (project/'players/models').glob('*/metadata.json')}
    exports = []
    try:
        for entry in journal:
            name = '%06d' % entry['update']
            latest = run_dir/'policies'/name
            if not latest.exists() or entry['update'] == journal[-1]['update']:
                export_policy(run_dir/'updates'/name, latest)
            if project is not None:
                identity = read_json(latest/'metadata.json')['id']
                public = published.get(identity, project/'players/models'/('policy-'+identity.split(':')[-1]))
                if not public.exists():
                    export_policy(run_dir/'updates'/name, public)
                    published[identity] = public
                exports.append((identity, latest, public))
        if project is not None:
            from .model_inventory import update_inventory
            update_inventory(project, run_dir, exports)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError('update boundary is committed; portable policy export failed; '
                           'resume latest to repair publication: %s' % exc) from exc
    return latest


def _snapshot(model, directory, update, role):
    path = directory / ('%06d.pt' % update)
    path.write_bytes(tensor_bytes(model.state_dict()))
    return Snapshot('%s-%06d' % (role, update), path, update, role)


def _stored_league(league, run_dir):
    """Rebind active references before deleting temporary raw snapshot sources."""
    history = league.history
    def stored(snapshot):
        digest = history[snapshot.id]['sha256']
        return replace(snapshot, weights_path=run_dir/'weights'/(digest+'.pt'))
    if isinstance(league, DynamicLeague):
        return DynamicLeague.restore(league.streams, tuple(stored(s) for s in league.snapshots),
                                     tuple(stored(s) for s in league.holdouts), history, league.state)
    return League.restore(league.streams, tuple(stored(s) for s in league.snapshots), league.config,
                          tuple(stored(s) for s in league.holdouts), league.history)


def _manifests(episodes, streams, notes):
    games = []
    for episode in episodes:
        choices = [[d.ply, d.seat, d.index] for d in episode.decisions]
        events = [[e.ply, e.actor, e.move_json, e.public_json] for e in episode.events]
        games.append({'game_id': episode.game_id, 'deal_seed': streams.seed('train.deal', episode.game_id),
                      'lineup': [asdict(entry) for entry in episode.lineup.entries],
                      'choice_sha256': hashlib.sha256(json_bytes(choices)).hexdigest(),
                      'events_sha256': hashlib.sha256(json_bytes(events)).hexdigest(),
                      'decisions': len(choices), 'plies': len(events),
                      'outcome': {'ranking': list(episode.result.ranking), 'vp': list(episode.result.vp),
                                  'strict_wins': strict_outcomes(episode.result.ranking).astype(int).tolist(),
                                  'tied_first': episode.result.ranking.count(0) > 1}})
    # Controller-only deal seeds and lineage; never exported to agents/policies.
    return {'train': games, 'dev': [], 'final': [], 'rng_restore_notes': list(notes)}


def _state(config, master, shuffle, game_id, update, league, metrics=(), manifests=None, best=None):
    return RunState(config=config, master_seed=master, controller_rng_states={'shuffle': shuffle.getstate()},
                    next_game_id=game_id, completed_updates=update, league_snapshots=league.snapshots,
                    holdouts=league.holdouts, league_history=league.history, development_best_id=best,
                    dynamic_league=league.state if isinstance(league, DynamicLeague) else None,
                    metrics=metrics, manifests=dict(manifests or {}, checkpoint_retention=ROLLING_POLICY))


def _train(config: TrainConfig, run_dir: Path, resume: Optional[Path] = None,
          max_updates: Optional[int] = None, *, stop_requested: Optional[Callable[[], bool]] = None,
          dynamic_members=None) -> TrainReport:
    """Train to an absolute update ceiling without changing the stored config.

    Wall time is a cooperative cutoff: reserve checkpoint_margin_seconds and the
    last observed update duration before launching another update. An unmeasured
    first update may overrun; in-flight work always finishes on graceful shutdown.
    budget_usd is launcher metadata; this local command never provisions compute.
    Optional development evaluation runs before the same atomic update commit.
    """
    if dynamic_members is not None and resume is None:
        raise ValueError("dynamic league activation requires a full historical checkpoint")
    if not isinstance(config, TrainConfig):
        raise ValueError('train requires a TrainConfig')
    ceiling = config.updates if max_updates is None else max_updates
    if type(ceiling) is not int or ceiling < 0:
        raise ValueError('max_updates must be a nonnegative absolute completed-update ceiling')
    run_dir = Path(run_dir).resolve()
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('run directory is not empty; use --resume latest or choose a new --out')
    started = time.monotonic()
    runtime_check(config.runtime.device)
    torch.set_num_threads(config.runtime.torch_threads)
    shuffle = random.Random(0)  # Private stream; restored after all construction.
    stop = stop_requested if stop_requested is not None else StopFlag()
    notes = ()
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.snapshots-', dir=str(run_dir)) as raw_dir:
        raw_dir = Path(raw_dir)
        if resume is None:
            master = secrets.randbits(64) if config.master_seed is None else config.master_seed
            streams = SeedStreams(master)
            random.seed(streams.seed('initialize.python'))
            np.random.seed(streams.seed('initialize.numpy') % 2**32)
            torch.manual_seed(streams.seed('initialize.torch'))
            shuffle.seed(streams.seed('shuffle'))
            components = ComponentData.load()
            model = BrassPolicy(config.model, components.schema, components).to(
                device=config.runtime.device, dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
            league = League(streams, _snapshot(model, raw_dir, 0, 'initial'), config.league)
            experiment = ensure_experiment(run_dir, config)
            state = _state(config, master, shuffle, 0, 0, league, manifests={'experiment_id': experiment['id'], 'replay_ids': []})
            latest = save_update(run_dir, model, optimizer, state)
            league = _stored_league(league, run_dir)
            for raw_path in raw_dir.iterdir():
                raw_path.unlink()
            state = replace(state, league_snapshots=league.snapshots, holdouts=league.holdouts)
        else:
            loaded = load_update(Path(resume), device=config.runtime.device, config=config)
            if loaded.path.parent.parent != run_dir:
                raise ValueError('resume checkpoint must belong to --out run directory')
            journal = committed_records(run_dir)
            if loaded.state.completed_updates != journal[-1]['update']:
                raise ValueError('resume requires the latest committed boundary; historical continuation would overwrite updates')
            state, model, latest = loaded.state, loaded.model, loaded.path
            optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
            optimizer.load_state_dict(loaded.optimizer_state)
            streams = SeedStreams(state.master_seed)
            league = restore_league(state)
            shuffle.setstate(state.controller_rng_states['shuffle'])
            notes = tuple(dict.fromkeys(loaded.rng_restore_notes +
                                       restore_rng_states(state.rng_states, device=config.runtime.device)))
            for note in notes:
                warnings.warn(note, RuntimeWarning)
        experiment = ensure_experiment(run_dir, config, state=state)
        from .main_bc import DESTINATION, source_parent
        project = project_for(run_dir)
        canonical = (project is not None and run_dir == project/DESTINATION and
                     experiment.get("parent") == source_parent())
        activation = run_dir/'dynamic-activation.json'
        first_activation = False
        effective = state
        if state.dynamic_league is None and (activation.exists() or dynamic_members is not None):
            from .league_activation import activation_league
            league = activation_league(run_dir, latest, state, streams, experiment['id'], dynamic_members)
            effective = replace(state, league_snapshots=league.snapshots, league_history=league.history,
                                dynamic_league=league.state)
            first_activation = not activation.exists()
        journal = rebuild_metrics(run_dir)
        latest_policy = publish_policies(run_dir, journal)
        maintain_rolling(run_dir, journal, state.development_best_id)
        restored = preflight(run_dir, experiment, effective, read_json(latest/'metadata.json')['id'],
                             resumed=resume is not None, notes=notes)
        print(json.dumps(restored, sort_keys=True), flush=True)
        last_duration = 0.
        reason = 'max_updates'
        while state.completed_updates < ceiling:
            if stop():
                reason = 'stop_requested'
                break
            wall = config.runtime.max_wall_time_seconds
            if wall is not None and time.monotonic()-started + last_duration + config.runtime.checkpoint_margin_seconds >= wall:
                reason = 'wall_time'
                break
            update_started = time.monotonic()
            collection_identity = read_json(latest/'metadata.json')['id']
            if (resume is not None and state.completed_updates == restored['last_completed_update']
                    and (canonical or first_activation or activation.exists())):
                from .league_activation import check_pending_training, seal_activation
                check_pending_training(league, state, experiment, collection_identity, first_activation)
                if first_activation:
                    seal_activation(run_dir, latest, effective, model, experiment['id'])
                    league = _stored_league(league, run_dir)
            collection_started = time.monotonic()
            with recording_context(category='training', experiment=experiment.get('legacy_replay_identity', experiment['id']),
                                   update=state.completed_updates + 1, candidate_id=collection_identity,
                                   experiment_path=str(run_dir.resolve())):
                episodes = collect_batch(model, league, streams, state.next_game_id,
                                         config.games_per_update, config.concurrent_games, config)
            collection_seconds = time.monotonic() - collection_started
            if [ep.game_id for ep in episodes] != list(range(state.next_game_id, state.next_game_id+config.games_per_update)):
                raise ValueError('collector must return all complete games in stable game ID order')
            # Bind played frozen seats before cadence can evict their snapshots.
            # History contains the validated immutable byte digest and role/update.
            frozen_ids = {entry.snapshot_id for episode in episodes for entry in episode.lineup.entries
                          if entry.kind == 'frozen'}
            frozen_identities = {}
            history = league.history
            for snapshot in league.snapshots:
                if snapshot.id in frozen_ids:
                    league.verify(snapshot)
                    entry = history[snapshot.id]
                    frozen_identities[snapshot.id] = replay_identity(snapshot.id, entry)
            if frozen_ids != frozen_identities.keys():
                raise ValueError('collected frozen replay snapshot is missing from the verified league')
            ppo_started = time.monotonic()
            metrics = update_policy(model, optimizer, episodes, config.ppo, shuffle)
            timings = {'collection': collection_seconds, 'ppo': time.monotonic() - ppo_started}
            if not all(math.isfinite(value) and value >= 0 for value in timings.values()):
                raise ValueError('collection/PPO timings must be finite and nonnegative; update was not published')
            metrics_data = asdict(metrics)
            if not all(math.isfinite(value) for value in metrics_data.values()):
                raise ValueError('PPO metrics must be finite; update was not published')
            update = state.completed_updates + 1
            if isinstance(league, DynamicLeague):
                for episode in episodes:
                    league.record_training(episode.game_id, episode.lineup, episode.result.ranking)
                if canonical:
                    snapshot = None
                    if update % league.next_stride(update) == 0:
                        raw = _snapshot(model, raw_dir, update, 'training')
                        snapshot = replace(raw, id=policy_id(model, update))
                    league.advance_main(update, snapshot)
            elif update == config.league.holdout_update:
                league.add(_snapshot(model, raw_dir, update, 'holdout'))
            elif update % config.league.snapshot_interval_updates == 0:
                league.add(_snapshot(model, raw_dir, update, 'training'))
            manifests = _manifests(episodes, streams, notes)
            record = {'update': update, 'first_game_id': state.next_game_id,
                      'next_game_id': state.next_game_id+len(episodes), 'games': len(episodes),
                      'ppo': metrics_data, 'timings_seconds': timings, 'pool_ids': [s.id for s in league.snapshots],
                      'holdout_ids': [s.id for s in league.holdouts], 'rng_restore_notes': list(notes)}
            best = state.development_best_id
            if config.eval.interval_updates and update % config.eval.interval_updates == 0:
                with recording_context(run_experiment=experiment['id']):
                    development = periodic_development(model, update, league, run_dir, config, journal)
                record['development'] = development
                manifests['dev'] = development['manifest']
                best = development['best_id']
            from brass_ai.replay_archive import Archive
            manifests['experiment_id'] = experiment['id']
            replay_experiment = experiment.get('legacy_replay_identity',experiment['id'])
            recordings = [r for r in Archive().records() if r['provenance'].get('update') == update and
                          (r['provenance'].get('experiment') == replay_experiment or
                           r['provenance'].get('run_experiment') == experiment['id'])]
            recorded_games = [r['provenance'].get('game') for r in recordings if r['category']=='training']
            if (len(set(recorded_games))!=len(recorded_games) or
                    not set(recorded_games)<={ep.game_id for ep in episodes}):
                raise ValueError('conflicting training recording IDs; update was not committed')
            manifests['replay_ids'] = [r['game_id'] for r in recordings]
            state = _state(config, state.master_seed, shuffle, state.next_game_id+len(episodes), update,
                           league, (record,), manifests, best)
            latest = save_update(run_dir, model, optimizer, state)
            league = _stored_league(league, run_dir)
            for raw_path in raw_dir.iterdir():
                raw_path.unlink()
            state = replace(state, league_snapshots=league.snapshots, holdouts=league.holdouts)
            journal = rebuild_metrics(run_dir)
            latest_policy = publish_policies(run_dir, journal)
            maintain_rolling(run_dir, journal, state.development_best_id)
            # Inspection files are disposable and never part of checkpoint truth.
            # Resume does not recollect committed games to repair missing replays.
            retain_training_replays(run_dir/'replays', episodes, streams,
                                    config.eval.training_replay_limit, collection_identity, frozen_identities)
            last_duration = time.monotonic()-update_started
            del episodes
        trace = tuple((game['game_id'], game['choice_sha256']) for entry in journal
                      for game in entry['manifests'].get('train', ()))
        state = replace(state, rng_states=capture_rng_states())
        return TrainReport(latest, latest_policy, state.completed_updates, state.next_game_id, model, state, trace, reason, notes, restored)


def train(config: TrainConfig, run_dir: Path, resume: Optional[Path] = None,
          max_updates: Optional[int] = None, *, stop_requested: Optional[Callable[[], bool]] = None,
          dynamic_members=None) -> TrainReport:
    with writer_lock(run_dir):
        return _train(config, run_dir, resume, max_updates, stop_requested=stop_requested,
                      dynamic_members=dynamic_members)
