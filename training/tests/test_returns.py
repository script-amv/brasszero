# SPDX-License-Identifier: Apache-2.0
"""Hand-computed terminal returns on each observer's own decision timeline."""
from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest

from brass_ai.arena import GameResult
from brass_ai.stats import is_win
from brass_rl.league import Lineup, LineupEntry
from brass_rl.rollout import Decision, Episode
from brass_rl.returns import episode_targets, seat_gae, strict_outcomes
from helpers import native_game


VALUES = np.array([[.2, .3, .1, .4], [.4, .2, .1, .3]], dtype=np.float32)
ADVANTAGES = [[.77, -.29, -.095, -.385], [.6, -.2, -.1, -.3]]
TARGETS = [[.97, .01, .005, .015], [1., 0., 0., 0.]]


def decision(ply, seat=0, values=(.2, .3, .1, .4), phase="ChooseAction"):
    game = native_game(phase, rotation=seat if phase == "ChooseAction" else 0)
    return Decision(7, ply, seat, game.observation(seat), 0, -.5, tuple(values))


def episode(decisions=(), ranking=(0, 1, 2, 3), lineup=None):
    # Target construction consumes only native observations, old values and result;
    # public-event memory reconstruction is deliberately outside this math fixture.
    return Episode(7, "{}", (), {}, tuple(decisions),
                   lineup or Lineup(tuple(LineupEntry("current", trainable=True) for _ in range(4))),
                   GameResult(1, ("a", "b", "c", "d"), tuple(ranking), (50, 40, 30, 20), 100))


def test_unique_win_and_tied_first():
    np.testing.assert_array_equal(strict_outcomes([2, 0, 1, 3]), [0, 1, 0, 0])
    np.testing.assert_array_equal(strict_outcomes([0, 0, 2, 3]), [0, 0, 0, 0])
    np.testing.assert_array_equal(strict_outcomes([0, 0, 0, 0]), [0, 0, 0, 0])
    assert strict_outcomes([2, 0, 1, 3]).dtype == np.float32


def test_outcomes_match_stage1_for_all_75_valid_ranking_patterns():
    rankings = {tuple(sum(other > score for other in scores) for score in scores)
                for scores in product(range(4), repeat=4)}
    assert len(rankings) == 75
    for ranking in rankings:
        result = episode(ranking=ranking).result
        np.testing.assert_array_equal(strict_outcomes(ranking), [is_win(result, s) for s in range(4)])


def test_two_decision_vector_gae():
    original = VALUES.copy()
    adv, targets = seat_gae(VALUES, np.array([1., 0., 0., 0.]), gamma=1., lam=.95)
    np.testing.assert_allclose(adv, ADVANTAGES, atol=1e-6)
    np.testing.assert_allclose(targets, TARGETS, atol=1e-6)
    np.testing.assert_array_equal(VALUES, original)
    assert adv.dtype == targets.dtype == np.float32


def test_terminal_has_zero_bootstrap_and_ties_have_no_reward():
    adv, targets = seat_gae(VALUES[-1:], np.zeros(4))
    np.testing.assert_allclose(adv, [[-.4, -.2, -.1, -.3]], atol=1e-6)
    np.testing.assert_array_equal(targets, [[0., 0., 0., 0.]])


def test_lambda_zero_uses_own_next_value():
    adv, targets = seat_gae(VALUES, [1, 0, 0, 0], lam=0.)
    np.testing.assert_allclose(adv, [[.2, -.1, 0., -.1], [.6, -.2, -.1, -.3]], atol=1e-6)
    np.testing.assert_allclose(targets, [[.4, .2, .1, .3], [1., 0., 0., 0.]], atol=1e-6)


def test_opponent_moves_not_bootstraps():
    first, last = decision(0), decision(4, values=VALUES[1])
    between = [decision(s, seat=s, values=(9., 8., 7., 6.)) for s in (1, 2, 3)]
    targets = episode_targets(episode([first] + between + [last]))
    np.testing.assert_allclose(targets[first.key].value_targets, TARGETS[0], atol=1e-6)
    np.testing.assert_allclose(targets[last.key].value_targets, TARGETS[1], atol=1e-6)
    assert targets[first.key].self_advantage == pytest.approx(.77, abs=1e-6)
    assert targets[last.key].self_advantage == pytest.approx(.6, abs=1e-6)
    assert targets[first.key].policy_mask


def test_same_seat_bootstraps_across_public_event_chunk_boundaries():
    first, last = decision(0), decision(129, values=VALUES[1])
    targets = episode_targets(episode([first, last]))
    np.testing.assert_allclose(targets[first.key].value_targets, TARGETS[0], atol=1e-6)
    assert targets[first.key].self_advantage == pytest.approx(.77, abs=1e-6)


@pytest.mark.parametrize("seat,want", [(0, [0, 0, 0, 1]), (1, [0, 0, 1, 0]),
                                       (2, [0, 1, 0, 0]), (3, [1, 0, 0, 0])])
def test_terminal_outcome_rotates_clockwise_to_observer_zero(seat, want):
    d = decision(0, seat=seat)
    target = episode_targets(episode([d], ranking=(1, 2, 3, 0)))[d.key]
    np.testing.assert_array_equal(target.value_targets, want)
    assert target.self_advantage == pytest.approx(.8 if seat == 3 else -.2, abs=1e-6)


def test_shortfall_owned_by_shortfall_seat():
    # Turn track begins at seat 0, but the shortfall actor is seat 2.
    d = decision(8, seat=2, phase={"Shortfall": {"player": 2, "owed": 3}})
    public = json.loads(d.observation.public_json())
    assert public["turn_order"][public["turn_idx"]] == 0
    assert public["phase"]["Shortfall"]["player"] == 2
    target = episode_targets(episode([d], ranking=(0, 1, 2, 3)))[d.key]
    assert target.self_advantage == pytest.approx(-.2, abs=1e-6)
    np.testing.assert_array_equal(target.value_targets, [0, 0, 1, 0])


def test_singletons_value_only():
    d = decision(8, seat=2, phase={"Shortfall": {"player": 2, "owed": 3}})
    assert d.observation.num_legal() == 1
    target = episode_targets(episode([d]))[d.key]
    assert not target.policy_mask
    np.testing.assert_array_equal(target.value_targets, [0, 0, 1, 0])


def test_empty_seat_trajectory():
    adv, targets = seat_gae(np.empty((0, 4)), [1, 0, 0, 0])
    assert adv.shape == targets.shape == (0, 4)
    assert adv.dtype == targets.dtype == np.float32
    assert episode_targets(episode()) == {}
    d = decision(0, seat=1)
    assert set(episode_targets(episode([d]))) == {d.key}


@pytest.mark.parametrize("kind,snapshot_id", [("frozen", "snapshot"), ("random", None), ("greedy", None)])
def test_non_current_decisions_reject_episode(kind, snapshot_id):
    lineup = Lineup((LineupEntry("current", trainable=True), LineupEntry(kind, snapshot_id),
                     LineupEntry("random"), LineupEntry("greedy")))
    with pytest.raises(ValueError, match="current"):
        episode_targets(episode([decision(0), decision(1, seat=1)], lineup=lineup))


@pytest.mark.parametrize("changes", [{"game_id": 8}, {"seat": -1}, {"ply": -1}])
def test_malformed_decision_identity_rejected(changes):
    with pytest.raises(ValueError):
        episode_targets(episode([replace(decision(0), **changes)]))


def test_duplicate_keys_and_reversed_own_timeline_rejected():
    d = decision(0)
    with pytest.raises(ValueError):
        episode_targets(episode([d, d]))
    with pytest.raises(ValueError):
        episode_targets(episode([decision(4), d]))


@pytest.mark.parametrize("values,outcome", [(np.zeros((2, 3)), [1, 0, 0, 0]),
                                            (np.zeros((2, 4)), [1, 0]),
                                            ([[np.nan, 0, 0, 0]], [1, 0, 0, 0])])
def test_invalid_vector_inputs_rejected(values, outcome):
    with pytest.raises(ValueError):
        seat_gae(values, outcome)


@pytest.mark.parametrize("ranking", [(1, 0, 2, 3), (0, 0, 2, 3)])
def test_terminal_critic_changes_only_value_labels(ranking):
    choices = [decision(i * 4 + s, seat=s) for i in range(2) for s in range(4)]
    game = episode(choices, ranking=ranking)
    control = episode_targets(game)
    direct = episode_targets(game, critic_target="terminal")
    for d in choices:
        assert control[d.key].self_advantage == direct[d.key].self_advantage
        assert control[d.key].policy_mask == direct[d.key].policy_mask
        np.testing.assert_array_equal(direct[d.key].value_targets, np.roll(strict_outcomes(ranking), -d.seat))


def test_score_reward_clamps_and_uses_each_seats_terminal_vp_with_gae():
    from brass_rl.returns import score_outcomes
    np.testing.assert_array_equal(score_outcomes([-1, 0, 200, 350]), [0, 0, 1, 1])
    d, e = decision(0, seat=1), decision(12, seat=1, values=VALUES[1])
    game = episode([d, e], ranking=(1, 0, 2, 3))
    game = replace(game, result=replace(game.result, vp=(200, 150, 0, -1)))
    targets = episode_targets(game, reward="score")
    adv, expected = seat_gae(VALUES, [.5, 0, 0, 1])
    for i, choice in enumerate((d, e)):
        np.testing.assert_array_equal(targets[choice.key].value_targets, expected[i])
        assert targets[choice.key].self_advantage == adv[i, 0]


def test_combined_direct_score_labels_keep_c_actor_and_rotate_every_position():
    from brass_rl.config import PPOConfig
    config = PPOConfig(reward="score", critic_target="terminal")
    choices = [decision(t*4+s, seat=s, values=VALUES[t]) for t in range(2) for s in range(4)]
    game = episode(choices)
    game = replace(game, result=replace(game.result, vp=(0, 150, 200, 240)))
    baseline = episode_targets(game, reward="score")
    combined = episode_targets(game, reward=config.reward, critic_target=config.critic_target)
    expected = ([0, .5, 1, 1], [.5, 1, 1, 0], [1, 1, 0, .5], [1, 0, .5, 1])
    for d in choices:
        target = combined[d.key]
        assert target.self_advantage == baseline[d.key].self_advantage
        assert target.policy_mask == baseline[d.key].policy_mask
        np.testing.assert_array_equal(target.value_targets, expected[d.seat])
    assert not np.array_equal(baseline[choices[0].key].value_targets, expected[0])
