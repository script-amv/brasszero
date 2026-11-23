# SPDX-License-Identifier: Apache-2.0
"""Atomic compact full-state forks. A profile supplies only its declared treatment."""
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Optional
import tempfile

import torch

from .artifacts import capture_rng_states, export_policy, load_update, restore_rng_states, save_update
from .config import TrainConfig
from .experiments import atomic, encoded, ensure_experiment, link_or_copy, parent_identity, same_state, sync_dir, writer_lock
from .policy_io import read_json
from .retention import ROLLING_POLICY
from .stability_checkpoint import independent_critic


@dataclass(frozen=True)
class ForkProfile:
    config: TrainConfig
    independent: bool
    manifest: dict
    experiment: dict
    loss_priority: Optional[dict] = None


def fork_learner(source, dest, configure):
    from .train import publish_policies

    source, dest = Path(source).resolve(), Path(dest).resolve()
    if not source.name.isdecimal() or source.parent.name != 'updates':
        raise ValueError('learner fork requires an explicit numeric checkpoint')
    if dest == source.parent.parent or source.parent.parent in dest.parents:
        raise ValueError('fork destination must be independent of the source run')
    before = capture_rng_states()
    with writer_lock(dest):
        try:
            if dest.exists():
                raise ValueError('fork destination already exists')
            parent = parent_identity(source)
            original = load_update(source, for_training=True)
            state, model = original.state, original.model
            plan = configure(original, source)
            with tempfile.TemporaryDirectory(prefix='.learner-fork-', dir=dest.parent) as temp:
                staging = Path(temp)/'run'
                staging.mkdir()
                meta = ensure_experiment(staging, plan.config, parent=parent)
                for snapshot in state.league_snapshots + state.holdouts:
                    target = staging/'weights'/snapshot.weights_path.name
                    if not target.exists():
                        link_or_copy(snapshot.weights_path, target)
                adam = torch.optim.Adam(model.parameters(), lr=plan.config.ppo.learning_rate)
                adam.load_state_dict(original.optimizer_state)
                critic = critic_adam = None
                clone = original.critic_state is None
                if plan.independent:
                    critic, critic_adam = independent_critic(original, clone=clone)
                base = replace(state, config=plan.config, metrics=(), development_best_id=None,
                               loss_priority=plan.loss_priority, manifests=dict(plan.manifest,
                               experiment_id=meta['id'], replay_ids=[], checkpoint_retention=ROLLING_POLICY,
                               source_manifest_sha256=parent['checkpoint_sha256']))
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
                if plan.independent:
                    expected_weights = model.state_dict() if clone else original.critic_state
                    expected_adam = critic_adam.state_dict() if clone else original.critic_optimizer_state
                    verified['critic_clone' if clone else 'critic_restored'] = same_state(expected_weights, child.critic_state)
                    verified['fresh_critic_adam' if clone else 'critic_adam_restored'] = same_state(
                        expected_adam, child.critic_optimizer_state)
                if plan.loss_priority is not None:
                    verified['loss_priority'] = child.state.loss_priority == plan.loss_priority
                if parent_identity(source) != parent or not all(verified.values()):
                    raise ValueError('source or fork state verification failed: '+str(verified))
                meta.update(plan.experiment, base_update=state.completed_updates, initial_next_game_id=state.next_game_id,
                            initial_state_verified=verified, source_resume_notes=list(original.rng_restore_notes),
                            parent_development_best_id=state.development_best_id,
                            historical_gaps=['Earlier updates, metrics and recordings are retained in the parent.'])
                atomic(staging/'experiment.json', encoded(meta))
                export_policy(path, staging/'policies'/path.name)
                sync_dir(staging)
                os.replace(staging, dest)
                sync_dir(dest.parent)
            publish_policies(dest)
            return dest/'updates'/source.name
        finally:
            restore_rng_states(before)
