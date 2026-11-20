# SPDX-License-Identifier: Apache-2.0
"""The public referee must never skip events, leak secrets, or change stateless games."""
import json
from dataclasses import FrozenInstanceError

import brass
import pytest

from brass_ai import arena


class Spy:
    name = "spy"

    def __init__(self, seat):
        self.bot = brass.RandomBot(seat)
        self.reset_seats = []
        self.initial = None
        self.events = []

    def reset(self, seat, initial_public_json):
        self.reset_seats.append(seat)
        self.initial = json.loads(initial_public_json)

    def observe_public(self, event):
        self.events.append(event)

    def act(self, obs):
        return self.bot.act(obs)


def spies():
    return [Spy(i) for i in range(4)]


def test_lifecycle_all_seats_once_per_applied_move():
    agents = spies()
    result = arena.play_game(agents, seed=0xDEADBEEF01020304)
    for seat, spy in enumerate(agents):
        assert spy.reset_seats == [seat]
        assert [event.ply for event in spy.events] == list(range(1, result.moves + 1))
        assert json.loads(spy.events[-1].public_json)["game_over"] is True
        assert spy.events == agents[0].events
    with pytest.raises(FrozenInstanceError):
        agents[0].events[0].ply = 99


@pytest.mark.parametrize("index", [True, "0", 0.0, -1, 999999])
def test_illegal_apply_emits_no_event(index):
    agents = spies()
    referee = arena.RefereeGame(13, agents)
    seat, obs = referee.pending()
    before = obs.to_json()
    with pytest.raises(arena.AgentError, match="seat %d.*ply 0.*invalid move index" % seat):
        referee.apply_index(index)
    assert referee.pending()[1].to_json() == before
    assert all(a.events == [] for a in agents)


@pytest.mark.parametrize("hook,ply", [("reset", 0), ("observe_public", 1)])
def test_stateful_hook_errors_name_seat_and_ply(hook, ply):
    agents = spies()
    def fail(*args):
        raise LookupError("hook failed")
    setattr(agents[2], hook, fail)
    with pytest.raises(arena.AgentError, match=r"seat 2.*spy.*ply %d.*%s" % (ply, hook)) as ei:
        referee = arena.RefereeGame(13, agents)
        referee.apply_index(0)
    assert isinstance(ei.value.__cause__, LookupError)


@pytest.mark.parametrize("hook", ["reset", "observe_public"])
def test_lifecycle_rejects_partial_hooks(hook):
    agents = [brass.RandomBot(i) for i in range(4)]
    class Partial:
        name = "partial"
        def act(self, obs):
            return 0
    agent = Partial()
    setattr(agent, hook, lambda *args: None)
    agents[1] = agent
    with pytest.raises(ValueError, match="seat 1.*both.*reset.*observe_public"):
        arena.RefereeGame(13, agents)


def test_events_exclude_secrets():
    agents = spies()
    seed = 0xDEADBEEF01020304
    referee = arena.RefereeGame(seed, agents)
    initial = agents[0].initial
    full = json.loads(brass.Game(seed).to_json())
    for seat, p in enumerate(initial["players"]):
        assert p["hand_size"] == 8
        assert p["discard"] == []  # in particular, never full.players[seat].discard[0]
        assert full["players"][seat]["discard"]
        assert p["has_face_down_discard"] is True
        assert p["wild_location_in_hand"] == p["wild_industry_in_hand"] == 0
    event = referee.apply_index(0)
    forbidden = {"my_hand", "hand", "hands", "deck", "rng_seed", "reshuffles", "seed", "setup", "discard_bottoms"}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(initial)
    check(json.loads(event.public_json))
    assert json.loads(event.move_json) == json.loads(brass.Game(seed).legal_moves()[0])
    assert json.loads(event.public_json) == json.loads(referee.pending()[1].public_json())


def test_referee_lifecycle_boundaries_and_preapply_callback():
    referee = arena.RefereeGame(13, spies())
    with pytest.raises(RuntimeError, match="not over"):
        referee.result()
    while not referee.is_over():
        seat, obs = referee.pending()
        assert seat == obs.me
        referee.apply_index(0)
    assert referee.result().moves > 0
    with pytest.raises(RuntimeError, match="over"):
        referee.pending()
    with pytest.raises(RuntimeError, match="over"):
        referee.apply_index(0)
    game = brass.Game(13)
    before = game.observation(game.current_player()).to_json()
    calls = []
    arena.play_game(spies(), 13, on_move=lambda seat, obs, index: calls.append(obs.to_json()))
    assert calls[0] == before


def test_stateless_match_unchanged():
    # Recorded from the old play_game before refactoring, including complete results.
    actual = arena.run_match(["random"] * 4, deals=1, master_seed=3)
    expected = [((1, 2, 0, 3), (2, 0, 4, 0), 157),
                ((1, 2, 2, 0), (2, 0, 0, 2), 157),
                ((0, 2, 3, 1), (15, 1, 0, 6), 168),
                ((3, 0, 0, 0), (2, 2, 2, 2), 155)]
    assert [(r.ranking, r.vp, r.moves) for r in actual] == expected
    assert all(r.seed == 12907177968877551943 and r.seats == ("random",) * 4 for r in actual)


def test_lifecycle_rejects_noncallable_hooks():
    agents = spies()
    agents[1].reset = None
    agents[1].observe_public = None
    with pytest.raises(ValueError, match="seat 1.*both callable"):
        arena.RefereeGame(13, agents)


def test_hook_failure_still_broadcasts_applied_move_to_other_seats():
    agents = spies()
    def fail(event):
        raise LookupError("failed after apply")
    agents[1].observe_public = fail
    referee = arena.RefereeGame(13, agents)
    before = referee.pending()[1].public_json()
    with pytest.raises(arena.AgentError, match="seat 1.*ply 1"):
        referee.apply_index(0)
    assert referee.pending()[1].public_json() != before
    for seat in (0, 2, 3):
        assert len(agents[seat].events) == 1
        assert agents[seat].events[0].ply == 1


def test_referee_move_limit_stops_before_another_apply(monkeypatch):
    monkeypatch.setattr(arena, "MAX_MOVES", 1)
    agents = spies()
    referee = arena.RefereeGame(13, agents)
    referee.apply_index(0)
    with pytest.raises(RuntimeError, match="exceeded 1 moves"):
        referee.apply_index(0)
    assert all(len(a.events) == 1 for a in agents)


@pytest.mark.parametrize("count", [0, 3, 5])
def test_referee_requires_four_agents(count):
    with pytest.raises(ValueError, match="exactly 4"):
        arena.RefereeGame(13, [brass.RandomBot(i) for i in range(count)])


def test_public_events_preserve_scout_wild_counts():
    referee = arena.RefereeGame(13, spies())
    actor, obs = referee.pending()
    index = next(i for i, text in enumerate(obs.legal_moves()) if "Scout" in json.loads(text))
    event = referee.apply_index(index)
    player = json.loads(event.public_json)["players"][actor]
    assert player["wild_location_in_hand"] == player["wild_industry_in_hand"] == 1
    assert player["hand_size"] == 8  # End of the first Canal turn refills the hand.
    assert "hand" not in player
