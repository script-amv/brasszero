# SPDX-License-Identifier: Apache-2.0
import pytest

from brass_ai.arena import GameResult, run_match
from brass_ai.stats import is_win, summarize


def R(ranking, vp, seats=("a", "b", "c", "d")):
    return GameResult(seed=1, seats=tuple(seats), ranking=tuple(ranking), vp=tuple(vp), moves=150)


def test_tie_for_first_is_not_a_win():
    r = R([0, 0, 2, 3], [50, 50, 40, 30])
    assert not is_win(r, 0) and not is_win(r, 1)
    s = {x.name: x for x in summarize([r])}
    assert s["a"].wins == 0 and s["a"].vp_margin == 0


def test_summary_values():
    rs = [R([0, 1, 2, 3], [60, 50, 40, 30])] * 10
    a = {x.name: x for x in summarize(rs)}["a"]
    assert a.wins == 10 and a.win_rate == 1.0 and a.mean_rank == 1.0 and a.vp_margin == 10
    assert 0.69 < a.win_ci[0] < 0.73 and a.win_ci[1] == 1.0  # Wilson n=10, p=1


def test_same_spec_in_several_seats():
    from brass_ai.rating import fit_ratings

    r = R([0, 1, 2, 3], [60, 50, 40, 30], seats=("greedy", "random", "random", "random"))
    s = {x.name: x for x in summarize([r] * 4)}
    assert s["random"].seat_games == 12 and s["greedy"].seat_games == 4
    assert set(fit_ratings([r] * 4)) == {"greedy", "random"}


def test_single_game_ci_collapses_and_sorted():
    s = summarize([R([0, 1, 2, 3], [60, 50, 40, 30])])
    assert s[0].name == "a" and s[0].rank_ci == (1.0, 1.0)
    assert [x.win_rate for x in s] == sorted((x.win_rate for x in s), reverse=True)


def test_sample_sd_used_for_ci():
    rs = [R([0, 1, 2, 3], [60, 50, 40, 30]), R([1, 0, 2, 3], [50, 60, 40, 30])]
    a = {x.name: x for x in summarize(rs)}["a"]
    # ranks 1,2: mean 1.5, sample sd = 0.7071; half-width = 1.96*0.7071/sqrt(2) = 0.98
    assert a.mean_rank == 1.5
    assert a.rank_ci[0] == pytest.approx(1.5 - 0.98, abs=1e-9)
    assert a.rank_ci[1] == pytest.approx(1.5 + 0.98, abs=1e-9)


@pytest.mark.slow
def test_greedy_clearly_beats_random():
    s = {x.name: x for x in summarize(run_match(["greedy", "random", "random", "random"], 25, 1))}
    assert s["greedy"].win_ci[0] > 0.5
