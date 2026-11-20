# SPDX-License-Identifier: Apache-2.0
import brass
import pytest

from brass_ai import agents as agents_mod
from brass_ai import arena
from brass_ai.agents import make_agent
from brass_ai.arena import AgentError, play_game, run_match
from brass_ai.results import read_results, write_results
from splitmix_ref import unsplitmix64


def test_run_match_rotates_seats():
    rs = run_match(["greedy", "random", "random", "random"], deals=2, master_seed=7)
    assert len(rs) == 8
    for d in range(2):
        block = rs[4 * d: 4 * d + 4]
        assert len({r.seed for r in block}) == 1
        assert sorted(r.seats.index("greedy") for r in block) == [0, 1, 2, 3]


def test_run_match_reproducible():
    a = run_match(["random"] * 4, deals=1, master_seed=3)
    assert a == run_match(["random"] * 4, deals=1, master_seed=3)


def test_run_match_needs_four_specs():
    with pytest.raises(ValueError):
        run_match(["random"] * 3, deals=1, master_seed=1)


def test_make_agent_unknown_lists_known():
    with pytest.raises(KeyError, match="random"):
        make_agent("nope", 1)


class Bad:
    name = "bad"

    def __init__(self, fn):
        self.fn = fn

    def act(self, obs):
        return self.fn(obs)


@pytest.mark.parametrize("fn", [lambda o: o.num_legal(), lambda o: True, lambda o: "0"])
def test_bad_agent_index_raises_agent_error(fn):
    agents = [Bad(fn)] + [make_agent("random", 0x5EED_0000_0000 + i) for i in range(3)]
    with pytest.raises(AgentError, match="seat 0.*bad"):
        play_game(agents, 11)


def test_agent_exception_is_wrapped():
    agents = [Bad(lambda o: 1 / 0)] + [make_agent("random", i) for i in range(3)]
    with pytest.raises(AgentError, match="seat 0.*bad") as ei:
        play_game(agents, 11)
    assert isinstance(ei.value.__cause__, ZeroDivisionError)


def test_on_move_called_per_move():
    calls = []
    agents = [make_agent("random", i) for i in range(4)]
    r = play_game(agents, 5, on_move=lambda p, obs, i: calls.append((p, i)))
    assert len(calls) == r.moves and r.seats == ("random",) * 4


def test_results_round_trip(tmp_path):
    rs = run_match(["random"] * 4, deals=1, master_seed=5)
    p = tmp_path / "r.jsonl"
    write_results(str(p), rs)
    write_results(str(p), rs)
    assert read_results([str(p)]) == rs + rs


def test_agent_seeds_not_derivable_into_deal_seeds(monkeypatch):
    seen = []

    def spy(spec, seed):
        seen.append(seed)
        return make_agent(spec, seed)

    monkeypatch.setattr(arena, "make_agent", spy)
    rs = run_match(["random"] * 4, deals=2, master_seed=7)
    deals = {r.seed for r in rs}
    assert len(seen) == 2 * 4 * 4 and len(set(seen)) == len(seen)
    for a in seen:
        inv = unsplitmix64(a)
        candidates = {a, inv}
        for r in range(4):
            for s in range(4):
                candidates.add(inv ^ (r << 8) ^ s)
                candidates.add(a ^ (r << 8) ^ s)
        assert not candidates & deals
    deal_json = {brass.Game(d).to_json() for d in deals}
    for a in seen[:4]:
        for x in (a, unsplitmix64(a)):
            assert brass.Game(x).to_json() not in deal_json


def test_run_match_labels_seats_by_spec(monkeypatch):
    monkeypatch.setitem(agents_mod._REGISTRY, "greedy-b", brass.GreedyBot)
    assert make_agent("greedy-b", 1).name == "greedy"
    rs = run_match(["greedy-b", "greedy", "random", "random"], deals=1, master_seed=3)
    for r in rs:
        assert sorted(r.seats) == ["greedy", "greedy-b", "random", "random"]


def test_play_game_labels_override_names():
    agents = [make_agent("random", i) for i in range(4)]
    r = play_game(agents, 5, labels=("a", "b", "c", "d"))
    assert r.seats == ("a", "b", "c", "d")
