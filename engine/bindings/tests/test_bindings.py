# SPDX-License-Identifier: Apache-2.0
import json
import random

import pytest

import brass


def play(seed):
    g = brass.Game(seed)
    rng = random.Random(seed)
    n = 0
    while not g.is_over():
        assert g.current_player() in range(4)
        g.apply(rng.choice(g.legal_moves()))
        n += 1
        assert n < 100000
    return g


def test_random_game_and_ranking():
    g = play(1)
    assert g.current_player() is None
    r = g.ranking()
    assert len(r) == 4 and all(0 <= x < 4 for x in r)
    assert min(r) == 0
    # permutation-with-ties: rank == number of players strictly ahead
    for x in r:
        assert x == sum(1 for y in r if y < x)


def _moves_from(seed, n):
    g = brass.Game(seed)
    rng = random.Random(seed)
    for _ in range(n):
        g.apply(rng.choice(g.legal_moves()))
    return g


def test_illegal_move_raises():
    g = brass.Game(2)
    with pytest.raises(ValueError):
        g.apply("not json")
    # a move legal a few plies ahead but not now
    ahead = _moves_from(2, 6)
    now = set(g.legal_moves())
    bad = next(m for m in ahead.legal_moves() if m not in now)
    before = g.to_json()
    with pytest.raises(ValueError):
        g.apply(bad)
    assert g.to_json() == before


def test_json_round_trip_and_determinism():
    g = _moves_from(7, 20)
    s = g.to_json()
    assert brass.Game.from_json(s).to_json() == s
    assert _moves_from(7, 20).to_json() == s


def test_from_json_corrupt_raises():
    d = json.loads(brass.Game(1).to_json())
    d["deck"] = d["deck"][1:]
    with pytest.raises(ValueError):
        brass.Game.from_json(json.dumps(d))
    with pytest.raises(ValueError):
        brass.Game.from_json("{}")


def test_clone_independent():
    g = brass.Game(3)
    c = g.clone()
    before = c.to_json()
    g.apply(g.legal_moves()[0])
    assert c.to_json() == before
    assert g.to_json() != before


def test_determinize_preserves_observation():
    g = _moves_from(9, 10)
    p = g.current_player()
    d = g.determinize(p, 123)
    assert json.loads(d.observe(p)) == json.loads(g.observe(p))
    assert not d.is_over()


def test_observe_bad_player():
    with pytest.raises(ValueError):
        brass.Game(1).observe(4)


def test_observe_hides_hands():
    g = brass.Game(5)
    full = json.loads(g.to_json())
    obs = json.loads(g.observe(1))
    assert obs["me"] == 1
    assert "my_hand" in obs and "players" not in obs and "deck" not in obs
    assert "deck" not in obs["public"]
    for pp in obs["public"]["players"]:
        assert "hand" not in pp
        assert pp["has_face_down_discard"] is True  # Canal Era
    # the face-down setup discard (engine discard[0]) is hidden from everyone
    for q in range(4):
        assert len(obs["public"]["players"][q]["discard"]) == len(full["players"][q]["discard"]) - 1
    p0_hand = full["players"][0]["hand"]
    assert obs["my_hand"] == full["players"][1]["hand"]
    assert obs["my_hand"] != p0_hand
    assert obs["hand_sizes"][0] == len(p0_hand)
    assert obs["deck_size"] == len(full["deck"])
    # the observation never serializes player 0's hand as a list
    assert json.dumps(p0_hand) not in g.observe(1)
    if g.current_player() != 1:
        assert obs["legal_moves"] == []


def play_partial(seed, n):
    g = brass.Game(seed)
    rng = random.Random(seed)
    for _ in range(n):
        if g.is_over():
            break
        g.apply(rng.choice(g.legal_moves()))
    return g


def hidden_twin(g, p):
    """Same game with an opponent's non-wild hand card swapped with the top deck card."""
    s = json.loads(g.to_json())
    q = (p + 1) % 4
    i = next(i for i, c in enumerate(s["players"][q]["hand"]) if c not in ("wild-location", "wild-industry"))
    s["players"][q]["hand"][i], s["deck"][-1] = s["deck"][-1], s["players"][q]["hand"][i]
    assert json.dumps(s) != g.to_json()
    return brass.Game.from_json(json.dumps(s))


def test_observation_hides_hidden_cards():
    g = play_partial(11, 20)  # random moves from a seed (deck still non-empty)
    p = g.current_player()
    h = hidden_twin(g, p)
    assert g.observation(p).to_json() == h.observation(p).to_json()
    assert g.observation(p).determinize(5).to_json() == h.observation(p).determinize(5).to_json()


def test_observation_exposes_no_game_state():
    public = {n for n in dir(brass.Observation) if not n.startswith("_")}
    assert public == {
        "me",
        "legal_moves",
        "num_legal",
        "to_json",
        "public_json",
        "from_json",
        "determinize",
        "render",
        "describe_move",
    }


def test_bots_play_full_game_by_index():
    g, bots = brass.Game(0xC0FFEE), [brass.GreedyBot(1), brass.RandomBot(2), brass.RandomBot(3), brass.RandomBot(4)]
    while not g.is_over():
        p = g.current_player()
        o = g.observation(p)
        assert o.legal_moves() == g.legal_moves()
        g.apply_index(bots[p].act(o))
    assert sorted(g.ranking())[0] == 0 and len(g.vp()) == 4


def test_greedy_scores_match_legal_moves():
    g = brass.Game(3)
    o = g.observation(g.current_player())
    assert len(brass.GreedyBot(0).scores(o)) == o.num_legal()


def test_apply_index_out_of_range_raises():
    g = brass.Game(3)
    with pytest.raises(ValueError):
        g.apply_index(len(g.legal_moves()))


def test_render_and_describe():
    g = brass.Game(3)
    o = g.observation(g.current_player())
    assert isinstance(o.render(), str) and o.render()
    assert isinstance(o.describe_move(0), str)
    with pytest.raises(IndexError):
        o.describe_move(o.num_legal())


def test_observation_public_json_is_only_public_snapshot():
    obs = brass.Game(5).observation(1)
    assert json.loads(obs.public_json()) == json.loads(obs.to_json())["public"]


def _assert_transport_round_trip(obs):
    clone = brass.Observation.from_json(obs.to_json())
    assert clone.to_json() == obs.to_json()
    assert clone.legal_moves() == obs.legal_moves()
    assert clone.public_json() == obs.public_json()
    assert clone.render() == obs.render()
    for seed in (0, 17, 2**64 - 1):
        game = clone.determinize(seed)
        assert game.observation(obs.me).to_json() == obs.to_json()
    if clone.num_legal():
        assert clone.describe_move(0) == obs.describe_move(0)


def test_observation_transport_initial_late_terminal_and_nonacting():
    g = brass.Game(9)
    rng = random.Random(9)
    for ply in range(1000):
        if ply in (0, 20, 90) or g.is_over():
            for seat in range(4):
                _assert_transport_round_trip(g.observation(seat))
        if g.is_over():
            break
        g.apply_index(rng.randrange(len(g.legal_moves())))
    assert g.is_over()


def test_observation_transport_shortfall():
    # Build a conservation-valid state with a genuine shortfall decision (as engine tests do).
    d = json.loads(brass.Game(11).to_json())
    d["phase"] = {"Shortfall": {"player": 2, "owed": 3}}
    d["players"][2]["money"] = 0
    d["players"][2]["income_space"] = 7
    d["slots"][0] = {"owner": 2, "industry": "cotton", "level": 1, "flipped": False, "resources": 0}
    d["players"][2]["mat_remaining"][0][0] -= 1
    g = brass.Game.from_json(json.dumps(d))
    for seat in range(4):
        _assert_transport_round_trip(g.observation(seat))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(seed=1),
    lambda d: d["public"].update(deck=[]),
    lambda d: d["public"]["players"][0].update(hand=[]),
    lambda d: d["legal_moves"][0][next(iter(d["legal_moves"][0]))].update(secret=3),
    lambda d: d.update(me=4),
    lambda d: d.update(my_hand=[]),
    lambda d: d["my_hand"].__setitem__(0, {"location": 255}),
    lambda d: d["public"]["players"][0]["mat_remaining"][0].__setitem__(7, 1),
    lambda d: d["hand_sizes"].__setitem__(0, 7),
    lambda d: d.update(deck_size=0),
    lambda d: d["public"]["players"][1].update(wild_industry_in_hand=255),
    lambda d: d["public"]["players"][0].update(income_space=100),
    lambda d: d["public"]["links"].__setitem__(0, 4),
    lambda d: d["public"]["slots"].append(None),
    lambda d: d["public"]["merchant_beer"].pop(),
    lambda d: d["public"].update(turn_order=[0, 0, 2, 3]),
    lambda d: d["public"].update(turn_idx=4),
    lambda d: d.update(phase="GameOver"),
    lambda d: d["public"].update(game_over=True),
    lambda d: d["legal_moves"].__setitem__(0, {"Canal": {"card": d["my_hand"][0], "link": 255}}),
    lambda d: d["legal_moves"].reverse(),
    lambda d: d["legal_moves"].pop(),
])
def test_observation_transport_rejects_malformed_or_inconsistent(mutation):
    game = brass.Game(3)
    d = json.loads(game.observation(game.current_player()).to_json())
    mutation(d)
    with pytest.raises(ValueError):
        brass.Observation.from_json(json.dumps(d))


def test_observation_transport_rejects_full_state_nonacting_moves_and_bad_tiles():
    with pytest.raises(ValueError):
        brass.Observation.from_json(brass.Game(3).to_json())
    game = brass.Game(3)
    d = json.loads(game.observation((game.current_player() + 1) % 4).to_json())
    d["legal_moves"] = ["EndAction"]
    with pytest.raises(ValueError):
        brass.Observation.from_json(json.dumps(d))
    for tile in [
        {"owner": 4, "industry": "cotton", "level": 1, "flipped": False, "resources": 0},
        {"owner": 0, "industry": "cotton", "level": 255, "flipped": False, "resources": 0},
        {"owner": 0, "industry": "cotton", "level": 1, "flipped": False, "resources": 255},
    ]:
        d["legal_moves"] = []
        d["public"]["slots"][0] = tile
        with pytest.raises(ValueError):
            brass.Observation.from_json(json.dumps(d))


def test_component_data_matches_engine():
    from pathlib import Path
    data = json.loads(brass.component_data())
    # The reference fixture is derived exclusively from the Rust component tables.
    board = json.loads((Path(__file__).parents[3] / "engine/bindings/tests/fixtures/components.json").read_text())
    for key in board:
        assert data[key] == board[key]
    assert data["version"] == 1
    fingerprint = data["content_hash"]
    payload = {k: v for k, v in data.items() if k != "content_hash"}
    # Independently recompute the documented non-security content fingerprint.
    h = 0xcbf29ce484222325
    for byte in json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"):
        h = ((h ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    assert fingerprint == "fnv1a64:%016x" % h
    assert brass.component_data() == brass.component_data()
    assert len(data["locations"]) == 27 and len(data["links"]) == 39
    assert len(data["slots"]) == len(json.loads(brass.Game(1).observation(0).public_json())["slots"])
    assert data["links"][29] == {"a": 14, "b": 15, "extra": 21, "canal": True, "rail": True}
    assert len(data["deck_4p"]) == 64
    assert sum(t["count"] for t in data["tiles"]) == 45
    assert data["constants"]["START_MONEY"] == 17
    assert len(data["merchant_tiles_4p"]) == 9


def test_observation_transport_rejects_duplicate_known_fields():
    obs = brass.Game(3).observation(0)
    text = obs.to_json()
    # A JSON tree parser alone drops duplicate keys before schema validation.
    duplicate = text.replace('"me":0', '"me":0,"me":0', 1)
    assert duplicate != text
    with pytest.raises(ValueError):
        brass.Observation.from_json(duplicate)


@pytest.mark.parametrize("change", ["round", "empty_actor", "shortfall_without_tile"])
def test_observation_transport_rejects_inconsistent_flow(change):
    game = brass.Game(3)
    d = json.loads(game.observation(game.current_player()).to_json())
    if change == "round":
        d["public"]["round"] = 9  # 64-card 4-player eras cannot reach round 9.
    elif change == "empty_actor":
        actor = d["me"]
        # Move this viewer's cards back to the unseen deck, preserving conservation.
        d["deck_size"] += len(d["my_hand"])
        d["public"]["deck_size"] = d["deck_size"]
        d["my_hand"] = []
        d["hand_sizes"][actor] = 0
        d["public"]["players"][actor]["hand_size"] = 0
        d["legal_moves"] = []
    else:
        actor = d["me"]
        d["phase"] = d["public"]["phase"] = {"Shortfall": {"player": actor, "owed": 3}}
        d["public"]["players"][actor]["money"] = 0
        d["public"]["players"][actor]["income_space"] = 7
        d["legal_moves"] = []
    with pytest.raises(ValueError):
        brass.Observation.from_json(json.dumps(d))


@pytest.mark.parametrize("industry,level,slot,resources,flipped", [
    ("pottery", 5, 2, 0, False),  # Rail-only tile in Canal.
    ("coal", 1, 1, 0, False),  # Exhausted resource tile must have flipped.
])
def test_observation_transport_rejects_invalid_tile_domain(industry, level, slot, resources, flipped):
    d = json.loads(brass.Game(11).to_json())
    d["slots"][slot] = {"owner": 0, "industry": industry, "level": level,
                        "resources": resources, "flipped": flipped}
    industry_id = {"pottery": 2, "coal": 3}[industry]
    d["players"][0]["mat_remaining"][industry_id][level - 1] -= 1
    game = brass.Game.from_json(json.dumps(d))
    obs = game.observation(game.current_player())
    with pytest.raises(ValueError):
        brass.Observation.from_json(obs.to_json())
