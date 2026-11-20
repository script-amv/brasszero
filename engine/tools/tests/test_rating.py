# SPDX-License-Identifier: Apache-2.0
import math

from brass_ai.arena import GameResult
from brass_ai.rating import fit_ratings


def R(ranking, vp, seats=("a", "b", "c", "d")):
    return GameResult(seed=1, seats=tuple(seats), ranking=tuple(ranking), vp=tuple(vp), moves=150)


def test_rating_order_and_anchor():
    rs = [R([0, 1, 2, 3], [4, 3, 2, 1], seats=("x", "y", "z", "random"))] * 50
    rt = fit_ratings(rs)
    assert rt["random"] == 0 and rt["x"] > rt["y"] > rt["z"] > 0
    assert all(math.isfinite(v) for v in rt.values())


def test_rating_tie_is_draw():
    rs = [R([0, 0, 0, 0], [1, 1, 1, 1], seats=("p", "q", "random", "random"))] * 20
    rt = fit_ratings(rs)
    assert abs(rt["p"]) < 1e-6 and abs(rt["q"]) < 1e-6


def test_missing_anchor_uses_alphabetically_first():
    rs = [R([0, 1, 2, 3], [4, 3, 2, 1], seats=("x", "y", "z", "w"))] * 5
    rt = fit_ratings(rs)
    assert rt["w"] == 0 and rt["x"] > rt["y"] > rt["z"]
