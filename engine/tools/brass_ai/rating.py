# SPDX-License-Identifier: Apache-2.0
"""Bradley-Terry ratings fitted by the MM algorithm, reported on the Elo scale."""
import math
from typing import Dict, Sequence, Tuple

from .arena import GameResult


def fit_ratings(results: Sequence[GameResult], anchor: str = "random") -> Dict[str, float]:
    names = sorted({n for r in results for n in r.seats})
    if not names:
        return {}
    if anchor not in names:
        anchor = names[0]
    # wins[(i, j)]: score of i against j; games[frozenset]: pair game count (as ordered keys)
    score: Dict[Tuple[str, str], float] = {}
    for r in results:
        for a in range(4):
            for b in range(a + 1, 4):
                na, nb = r.seats[a], r.seats[b]
                if na == nb:
                    continue
                if r.ranking[a] < r.ranking[b]:
                    sa, sb = 1.0, 0.0
                elif r.ranking[a] > r.ranking[b]:
                    sa, sb = 0.0, 1.0
                else:
                    sa = sb = 0.5
                score[(na, nb)] = score.get((na, nb), 0.0) + sa
                score[(nb, na)] = score.get((nb, na), 0.0) + sb
    for n in names:  # virtual draw against the anchor keeps ratings finite
        if n != anchor:
            score[(n, anchor)] = score.get((n, anchor), 0.0) + 0.5
            score[(anchor, n)] = score.get((anchor, n), 0.0) + 0.5

    opponents = {n: [] for n in names}
    for (i, j) in score:
        opponents[i].append(j)
    p = {n: 1.0 for n in names}
    for _ in range(1000):
        new = {}
        for i in names:
            w = sum(score[(i, j)] for j in opponents[i])
            denom = sum((score[(i, j)] + score[(j, i)]) / (p[i] + p[j]) for j in opponents[i])
            new[i] = w / denom if denom > 0 and w > 0 else p[i]
        scale = new[anchor]
        new = {n: v / scale for n, v in new.items()}
        delta = max(abs(new[n] - p[n]) for n in names)
        p = new
        if delta < 1e-6:
            break
    return {n: (0.0 if n == anchor else 400 * math.log10(p[n])) for n in names}
