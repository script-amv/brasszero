# SPDX-License-Identifier: Apache-2.0
"""Per-agent summary statistics with 95% confidence intervals."""
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .arena import GameResult

Z = 1.96


@dataclass(frozen=True)
class Summary:
    name: str
    seat_games: int
    wins: int
    win_rate: float
    win_ci: Tuple[float, float]  # Wilson 95%
    mean_rank: float
    rank_ci: Tuple[float, float]
    vp_margin: float  # own VP minus the best other seat's VP
    margin_ci: Tuple[float, float]


def is_win(r: GameResult, seat: int) -> bool:
    """Strictly first after tiebreakers; a full tie for first is a win for nobody."""
    return r.ranking[seat] == 0 and sum(1 for x in r.ranking if x == 0) == 1


def _wilson(wins: int, n: int) -> Tuple[float, float]:
    p = wins / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    lo = 0.0 if wins == 0 else max(0.0, centre - half)
    hi = 1.0 if wins == n else min(1.0, centre + half)
    return (lo, hi)


def _mean_ci(xs: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, (mean, mean)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = Z * sd / math.sqrt(n)
    return mean, (mean - half, mean + half)


def summarize(results: Sequence[GameResult]) -> List[Summary]:
    wins: Dict[str, List[bool]] = {}
    ranks: Dict[str, List[float]] = {}
    margins: Dict[str, List[float]] = {}
    for r in results:
        for seat, name in enumerate(r.seats):
            wins.setdefault(name, []).append(is_win(r, seat))
            ranks.setdefault(name, []).append(r.ranking[seat] + 1)
            best_other = max(v for i, v in enumerate(r.vp) if i != seat)
            margins.setdefault(name, []).append(r.vp[seat] - best_other)
    out = []
    for name in wins:
        n = len(wins[name])
        w = sum(wins[name])
        mean_rank, rank_ci = _mean_ci(ranks[name])
        margin, margin_ci = _mean_ci(margins[name])
        out.append(
            Summary(
                name=name,
                seat_games=n,
                wins=w,
                win_rate=w / n,
                win_ci=_wilson(w, n),
                mean_rank=mean_rank,
                rank_ci=rank_ci,
                vp_margin=margin,
                margin_ci=margin_ci,
            )
        )
    out.sort(key=lambda s: -s.win_rate)
    return out
