# SPDX-License-Identifier: Apache-2.0
"""Strict terminal rewards and four-component GAE on complete own-seat timelines.

No intermediate reward, opponent-ply discount or privileged critic state is used.
Episode self advantages remain raw: the PPO batch consumer normalizes once over
all eligible non-singleton decisions across the entire rollout.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .rollout import Episode


@dataclass(frozen=True)
class DecisionTarget:
    """Observer-relative value targets, raw self advantage, policy eligibility."""

    value_targets: np.ndarray
    self_advantage: float
    policy_mask: bool


def strict_outcomes(ranking: Sequence[int]) -> np.ndarray:
    """One for the unique rank-zero seat, otherwise zero, in absolute seat order."""
    ranks = np.asarray(ranking)
    if (ranks.shape != (4,) or ranks.dtype.kind not in "iu"
            or np.any((ranks < 0) | (ranks > 3)) or not np.any(ranks == 0)):
        raise ValueError("ranking must contain four integer ranks in 0..3, including first")
    outcomes = np.zeros(4, dtype=np.float32)
    winners = np.flatnonzero(ranks == 0)
    if len(winners) == 1:
        outcomes[winners[0]] = 1.
    return outcomes


def score_outcomes(vp: Sequence[int]) -> np.ndarray:
    """Bounded terminal reward per absolute seat, computed after native scoring."""
    scores = np.asarray(vp, dtype=np.float64)
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise ValueError("score rewards require four finite VP values")
    return (1. - np.sqrt(1. - np.clip(scores / 200., 0., 1.))).astype(np.float32)


def seat_gae(values: np.ndarray, outcome_relative: np.ndarray,
             gamma: float = 1., lam: float = .95) -> Tuple[np.ndarray, np.ndarray]:
    """Return advantages/targets for one seat's successive decisions, shape [T,4].

    The last own decision receives the terminal outcome even if opponents act
    afterward. Earlier decisions bootstrap from that seat's next decision only;
    terminal bootstrap is zero. Inputs are not modified, including empty [0,4].
    """
    values = np.asarray(values, dtype=np.float32)
    outcome = np.asarray(outcome_relative, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4 or outcome.shape != (4,):
        raise ValueError("values must have shape [T,4] and outcome shape [4]")
    if not np.isfinite(values).all() or not np.isfinite(outcome).all():
        raise ValueError("values and outcome must be finite")
    if not (0. <= gamma <= 1.) or not (0. <= lam <= 1.):
        raise ValueError("gamma and lambda must be finite in [0,1]")
    advantages = np.empty_like(values)
    following_advantage = np.zeros(4, dtype=np.float32)
    next_value = np.zeros(4, dtype=np.float32)
    for t in range(len(values) - 1, -1, -1):
        reward = outcome if t == len(values) - 1 else 0.
        delta = reward + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * following_advantage
        following_advantage = advantages[t]
        next_value = values[t]
    return advantages, advantages + values


def episode_targets(episode: "Episode", *, gamma: float = 1.,
                    lam: float = .95, reward: str = "strict_win",
                    critic_target: str = "gae") -> Mapping[Tuple[int, int, int], DecisionTarget]:
    """Map Decision.key to raw targets from complete current-seat trajectories.

    Only current/trainable seats may appear in episode.decisions. A hand-built
    episode containing frozen/baseline decisions is rejected in its entirety,
    rather than contributing masked value examples. Singleton current decisions
    remain value examples with policy_mask=False. Empty seats yield no examples.
    Own-seat decisions must be ordered by increasing pre-event ply; opponents'
    decisions/events and recurrent chunk boundaries do not enter the recursion.
    """
    if reward not in ("strict_win", "score") or critic_target not in ("gae", "terminal"):
        raise ValueError("unknown reward or critic target")
    outcomes = score_outcomes(episode.result.vp) if reward == "score" else strict_outcomes(episode.result.ranking)
    by_seat = [[] for _ in range(4)]
    for decision in episode.decisions:
        if (decision.game_id != episode.game_id or type(decision.seat) is not int
                or not 0 <= decision.seat < 4 or type(decision.ply) is not int
                or decision.ply < 0):
            raise ValueError("decision identity must match episode and valid seat/ply")
        entry = episode.lineup.entries[decision.seat]
        if entry.kind != "current" or not entry.trainable:
            raise ValueError("episode decisions must belong to current/trainable seats")
        trajectory = by_seat[decision.seat]
        if trajectory and trajectory[-1].ply >= decision.ply:
            raise ValueError("own-seat decision plies must strictly increase (no duplicate keys)")
        trajectory.append(decision)

    result = {}  # type: Dict[Tuple[int, int, int], DecisionTarget]
    for seat, trajectory in enumerate(by_seat):
        if not trajectory:
            continue
        relative = np.roll(outcomes, -seat)
        advantages, targets = seat_gae([d.old_values for d in trajectory], relative, gamma=gamma, lam=lam)
        for i, decision in enumerate(trajectory):
            result[decision.key] = DecisionTarget(
                (relative if critic_target == "terminal" else targets[i]).copy(), float(advantages[i, 0]),
                decision.observation.num_legal() > 1)
    return result
