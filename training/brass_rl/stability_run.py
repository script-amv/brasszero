# SPDX-License-Identifier: Apache-2.0
"""Explicit compact forks and whole-update resume for the two local BC study arms."""
from dataclasses import replace
from pathlib import Path

from .config import TrainConfig
from .experiments import read_experiment, writer_lock
from .learner_fork import ForkProfile, fork_learner
from .train import _train


PROTOCOL = 'brass-ppo-stability/2'


def _verify_source(state):
    ppo = state.config.ppo
    if (state.manifests.get('stability_arm') is not None or state.dynamic_league is not None
            or state.config.games_per_update != 2 or state.config.runtime.device != 'cpu'
            or state.config.runtime.torch_threads != 1 or state.config.eval.interval_updates
            or (ppo.gamma, ppo.gae_lambda, ppo.epochs, ppo.target_kl, ppo.reward, ppo.critic_target)
            != (1., .95, 4, .03, 'score', 'terminal')):
        raise ValueError('stability fork requires the two-game CPU legacy league and declared BC PPO settings')


def fork_stability(source: Path, dest: Path, candidate: bool) -> Path:
    if type(candidate) is not bool:
        raise ValueError('stability fork requires a boolean candidate')
    arm = 'candidate' if candidate else 'control'

    def configure(original, source):
        _verify_source(original.state)
        cfg = replace(original.state.config, games_per_update=8 if candidate else 2)
        experiment = dict(mode='stability-study', stability_protocol=PROTOCOL, stability_arm=arm,
                          config_overrides={'games_per_update': {'from': 2, 'to': 8}} if candidate else {},
                          actor_sequences_per_step=4 if candidate else 1, critic_initialization=(
                              'exact source clone, independent public memory, fresh Adam' if candidate else 'shared source'),
                          kl_monitor=('fixed across-game sample after each actor step; full chronological KL after each actor epoch'
                                      if candidate else 'legacy minibatch'))
        return ForkProfile(cfg, candidate, {'stability_arm': arm}, experiment)

    return fork_learner(source, dest, configure)


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
