# SPDX-License-Identifier: Apache-2.0
"""Activate the supported BC profile from an explicitly named full boundary."""
from dataclasses import fields, replace
from functools import partial

from .config import BCConfig, BC_LEARNER, TrainConfig
from .experiments import read_experiment
from .learner_fork import ForkProfile, fork_learner
from .loss_priority import initial_exposure


def _profile(original, source, *, max_wall_time_seconds):
    state = original.state
    metadata = read_experiment(source.parent.parent)
    arm = state.manifests.get('stability_arm')
    if (metadata.get('stability_arm') != arm or metadata.get('learner') != state.config.learner
            or (arm and metadata.get('stability_protocol') != 'brass-ppo-stability/2')):
        raise ValueError('source experiment declaration differs from its learner checkpoint')
    if state.dynamic_league is not None or (source.parent.parent/'dynamic-activation.json').exists():
        raise ValueError('supported BC requires the sampled frozen BC league')
    cfg = state.config
    values = {field.name: getattr(cfg, field.name) for field in fields(TrainConfig)}
    values.update(ppo=replace(cfg.ppo, target_kl=.05), games_per_update=8)
    overrides = {}
    if max_wall_time_seconds is not None:
        values['runtime'] = replace(cfg.runtime, max_wall_time_seconds=max_wall_time_seconds)
        overrides['runtime.max_wall_time_seconds'] = {'from': cfg.runtime.max_wall_time_seconds, 'to': max_wall_time_seconds}
    current = BCConfig(**values)
    for name, old, new in [('learner', cfg.learner, BC_LEARNER), ('games_per_update', cfg.games_per_update, 8),
                           ('ppo.target_kl', cfg.ppo.target_kl, .05)]:
        if old != new:
            overrides[name] = {'from': old, 'to': new}
    exposure = state.loss_priority if state.loss_priority is not None else initial_exposure(state.next_game_id)
    experiment = dict(mode='bc-current', learner=BC_LEARNER, config_overrides=overrides,
                      critic_initialization='cloned' if original.critic_state is None else 'restored',
                      loss_priority_initialization='empty' if state.loss_priority is None else 'restored')
    return ForkProfile(current, True, {'learner': BC_LEARNER}, experiment, exposure)


def fork_bc(source, dest, *, max_wall_time_seconds=None):
    """Create a compact BC child, preserving the source actor, RNG, league and both existing Adams."""
    return fork_learner(source, dest, partial(_profile, max_wall_time_seconds=max_wall_time_seconds))
