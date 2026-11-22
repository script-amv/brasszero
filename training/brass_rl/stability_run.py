# SPDX-License-Identifier: Apache-2.0
"""Explicit compact forks and whole-update resume for the two local BC study arms."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import tempfile

import torch

from .artifacts import capture_rng_states, export_policy, load_update, restore_rng_states, save_update
from .config import TrainConfig
from .experiments import atomic, encoded, ensure_experiment, link_or_copy, read_experiment, sync_dir, writer_lock
from .learning_study import parent_identity, same_state
from .policy_io import read_json
from .retention import ROLLING_POLICY
from .train import _train, publish_policies


PROTOCOL = 'brass-ppo-stability/1'


def _verify_source(state):
    ppo = state.config.ppo
    if (state.manifests.get('stability_arm') is not None or state.dynamic_league is not None
            or state.config.games_per_update != 2 or state.config.runtime.device != 'cpu'
            or state.config.runtime.torch_threads != 1 or state.config.eval.interval_updates
            or (ppo.gamma, ppo.gae_lambda, ppo.epochs, ppo.target_kl, ppo.reward, ppo.critic_target)
            != (1., .95, 4, .03, 'score', 'terminal')):
        raise ValueError('stability fork requires the two-game CPU legacy league and declared BC PPO settings')


def fork_stability(source: Path, dest: Path, candidate: bool) -> Path:
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if type(candidate) is not bool or not source.name.isdecimal() or source.parent.name != 'updates':
        raise ValueError('stability fork needs an explicit numeric checkpoint and boolean candidate')
    if dest == source.parent.parent or source.parent.parent in dest.parents:
        raise ValueError('stability fork destination must be independent of the source run')
    arm = 'candidate' if candidate else 'control'
    before = capture_rng_states()
    with writer_lock(dest):
        try:
            if dest.exists():
                raise ValueError('stability fork destination already exists')
            # Numeric source envelopes are immutable. Verify the parent identity
            # around the entire fork without creating a lock or pin in its tree.
            parent = parent_identity(source)
            original = load_update(source)
            state = original.state
            _verify_source(state)
            cfg = replace(state.config, games_per_update=8 if candidate else 2)
            with tempfile.TemporaryDirectory(prefix='.stability-fork-', dir=dest.parent) as temp:
                staging = Path(temp)/'run'
                staging.mkdir()
                meta = ensure_experiment(staging, cfg, parent=parent)
                for snapshot in state.league_snapshots + state.holdouts:
                    target = staging/'weights'/snapshot.weights_path.name
                    if not target.exists():
                        link_or_copy(snapshot.weights_path, target)
                model = original.model
                adam = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)
                adam.load_state_dict(original.optimizer_state)
                critic = deepcopy(model) if candidate else None
                critic_adam = torch.optim.Adam(critic.parameters(), lr=cfg.ppo.learning_rate) if candidate else None
                # A compact child starts its own journal at this boundary. Earlier
                # metrics, selections and replays stay addressable in the named parent.
                base = replace(state, config=cfg, metrics=(), development_best_id=None, manifests={
                    'experiment_id': meta['id'], 'stability_arm': arm, 'replay_ids': [],
                    'checkpoint_retention': ROLLING_POLICY, 'source_manifest_sha256': parent['checkpoint_sha256']})
                path = save_update(staging, model, adam, base, inherited_policy=source, inherit_source_rng=True,
                                   critic=critic, critic_optimizer=critic_adam)
                child = load_update(path)
                verified = {
                    'actor': same_state(model.state_dict(), child.model.state_dict()),
                    'actor_adam': same_state(original.optimizer_state, child.optimizer_state),
                    'global_rng': same_state(state.rng_states, child.state.rng_states),
                    'shuffle_rng': same_state(state.controller_rng_states, child.state.controller_rng_states),
                    'league': state.league_history == child.state.league_history and
                              read_json(source/'manifest.json')['snapshots'] == read_json(path/'manifest.json')['snapshots'],
                    'counters_and_master_seed': (state.next_game_id, state.completed_updates, state.master_seed) ==
                        (child.state.next_game_id, child.state.completed_updates, child.state.master_seed),
                    'public_policy_bytes': all((source/n).read_bytes() == (path/n).read_bytes()
                                               for n in ('policy.pt', 'metadata.json'))}
                if candidate:
                    verified['critic_clone'] = same_state(model.state_dict(), child.critic_state)
                    verified['fresh_critic_adam'] = same_state(critic_adam.state_dict(), child.critic_optimizer_state)
                if parent_identity(source) != parent:
                    raise ValueError('source identity changed during stability fork')
                if not all(verified.values()):
                    raise ValueError('stability fork verification failed: '+str(verified))
                meta.update(mode='stability-study', stability_protocol=PROTOCOL, stability_arm=arm,
                            base_update=state.completed_updates, initial_next_game_id=state.next_game_id,
                            initial_state_verified=verified, source_resume_notes=list(original.rng_restore_notes),
                            parent_development_best_id=state.development_best_id,
                            historical_gaps=['Earlier updates, metrics and recordings are retained in the parent.'],
                            config_overrides={'games_per_update': {'from': 2, 'to': 8}} if candidate else {},
                            actor_sequences_per_step=4 if candidate else 1, critic_initialization=(
                                'exact source clone, independent public memory, fresh Adam' if candidate else 'shared source'),
                            kl_monitor='fixed across-game sample after each actor step' if candidate else 'legacy minibatch')
                atomic(staging/'experiment.json', encoded(meta))
                export_policy(path, staging/'policies'/path.name)
                sync_dir(staging)
                os.replace(staging, dest)
                sync_dir(dest.parent)
            publish_policies(dest)
            return dest/'updates'/source.name
        finally:
            restore_rng_states(before)


def train_stability(run: Path, additional_games: int, *, stop_requested=None):
    """Resume only the latest boundary. A graceful stop can return fewer whole updates."""
    run = Path(run).resolve()
    with writer_lock(run):
        meta = read_experiment(run)
        arm = meta.get('stability_arm')
        if meta.get('stability_protocol') != PROTOCOL or arm not in ('candidate', 'control'):
            raise ValueError('train_stability requires an explicit stability fork')
        cfg = TrainConfig.from_json(run/'config.json')
        if cfg.runtime.device != 'cpu' or cfg.runtime.torch_threads != 1:
            raise ValueError('the local stability protocol requires one CPU Torch thread')
        games = 8 if arm == 'candidate' else 2
        if cfg.games_per_update != games or type(additional_games) is not int or additional_games < 0 or additional_games % games:
            raise ValueError('additional_games must be a nonnegative whole multiple of the arm games per update')
        # Numeric directories, including an update whose pointer publication was
        # interrupted, are the authority. The run lock covers ceiling selection.
        update = max(int(p.name) for p in (run/'updates').iterdir() if p.is_dir() and p.name.isdecimal())
        return _train(cfg, run, run/'updates'/('%06d' % update), update + additional_games//games,
                      stop_requested=stop_requested, stability_arm=arm)
