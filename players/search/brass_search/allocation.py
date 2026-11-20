"""Root-only sequential halving, with transactional equal-allocation rounds."""
from dataclasses import dataclass
import math


class BudgetStop(Exception):
    """Only an ordinary cooperative cap; correctness errors must propagate."""


@dataclass(frozen=True)
class Candidate:
    key: str
    prior_score: float


@dataclass(frozen=True)
class Allocation:
    selected: str
    completed_rounds: int
    rollouts: int
    discarded_rollouts: int
    means: dict
    reason: str


def root_candidates(keys, logp, gumbels, count):
    if len(keys) != len(set(keys)) or not len(keys) == len(logp) == len(gumbels):
        raise ValueError('root candidates must have distinct keys and aligned scores')
    if any(not math.isfinite(x) for x in list(logp) + list(gumbels)):
        raise ValueError('root scores must be finite')
    candidates = [Candidate(k, p + g) for k, p, g in zip(keys, logp, gumbels)]
    return sorted(candidates, key=lambda c: (-c.prior_score, c.key))[:count]


def allocate(candidates, config, evaluate, fallback):
    """evaluate(move_key, global_world_index) shares indices across round survivors.

    A failed/unfinished round spends work but changes no Q or committed winner.
    Ordinary caps raise BudgetStop; all other exceptions propagate to the caller.
    """
    if config.max_rollouts == 0:
        return Allocation(fallback, 0, 0, 0, {}, 'zero_budget')
    if len(candidates) == 1:
        return Allocation(candidates[0].key, 0, 0, 0, {}, 'single_candidate')
    survivors = list(candidates)
    totals, counts, means = {}, {}, {}
    rounds, spent, world = 0, 0, 0
    selected = fallback
    while len(survivors) > 1:
        partial = {c.key: [] for c in survivors}
        before = spent
        try:
            for index in range(world, world + config.worlds_per_round):
                for candidate in sorted(survivors, key=lambda c: c.key):
                    if spent >= config.max_rollouts:
                        raise BudgetStop('rollout_cap')
                    # Count a started continuation even if its wall cap interrupts it.
                    spent += 1
                    value = evaluate(candidate.key, index)
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('rollout return must be finite in [0, 1]')
                    partial[candidate.key].append(value)
        except BudgetStop as exc:
            return Allocation(selected, rounds, spent, spent - before, dict(means), str(exc))
        for candidate in survivors:
            key = candidate.key
            totals[key] = totals.get(key, 0.) + math.fsum(partial[key])
            counts[key] = counts.get(key, 0) + config.worlds_per_round
            means[key] = totals[key] / counts[key]
        ranked = sorted(survivors, key=lambda c: (-(c.prior_score + config.value_scale * means[c.key]), c.key))
        selected = ranked[0].key
        survivors = ranked[:(len(ranked) + 1) // 2]
        rounds += 1
        world += config.worlds_per_round
    return Allocation(selected, rounds, spent, 0, means, 'completed')


def relative_to_absolute(values, actor):
    if len(values) != 4 or actor not in range(4) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('values must be four finite probabilities with a valid actor')
    return tuple(float(values[(q - actor) % 4]) for q in range(4))


def terminal_values(ranking):
    if len(ranking) != 4 or any(type(rank) is not int or rank not in range(4) for rank in ranking) or 0 not in ranking:
        raise ValueError('invalid native ranking')
    return tuple(float(rank == 0 and ranking.count(0) == 1) for rank in ranking)
