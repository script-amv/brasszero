# SPDX-License-Identifier: Apache-2.0
"""Semantic preservation tests; deleting any encoded field must change a check here."""
from dataclasses import fields, is_dataclass, replace
import copy
import hashlib
import json

import brass
import numpy as np
import pytest

from brass_ai.arena import PublicEvent
from brass_rl.schema import ComponentData
from brass_rl.features import encode_observation, encode_candidates, encode_public
from helpers import native_game, move_cases, case_move, wild_public_game, swap_private_hand_with_deck


@pytest.fixture(scope="module")
def components():
    return ComponentData.load()


def assert_same(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif is_dataclass(a):
        for field in fields(a):
            assert_same(getattr(a, field.name), getattr(b, field.name))
    else:
        assert a == b


def test_component_schema_is_canonical_and_complete(components):
    raw = json.loads(brass.component_data())
    assert components.data == raw
    assert components.component_hash.startswith("sha256:")
    assert components.schema.version == 1
    assert components.schema.component_hash == components.component_hash
    assert components.schema.fingerprint == ComponentData.load().schema.fingerprint
    assert len(components.schema.vocabularies["cards"]) == 27
    assert components.link_locations.shape == (39, 3)
    assert components.link_locations[29].tolist() == [14, 15, 21]
    assert components.link_location_mask[0].tolist() == [True, True, False]
    assert components.slot_locations[5] == 1
    assert components.merchant_locations.tolist() == [22, 23, 23, 24, 24, 25, 25, 26, 26]
    canonical = components.schema.canonical_json()
    assert components.schema.fingerprint == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    assert replace(components.schema, version=2).fingerprint != components.schema.fingerprint


@pytest.mark.parametrize("bad", [lambda: brass.Game(1), lambda: {},
                                      lambda: brass.Game(1).to_json()])
def test_encode_rejects_game(bad, components):
    with pytest.raises(TypeError, match="Observation"):
        encode_observation(bad(), components)


def test_hidden_swap_features_identical(components):
    game = native_game()
    view = game.observation(0)
    a, b = view.determinize(17), view.determinize(314)
    assert a.to_json() != b.to_json()
    assert_same(encode_observation(a.observation(0), components),
                encode_observation(b.observation(0), components))


def test_relative_seats(components):
    for phase in ("ChooseAction", {"Shortfall": {"player": 0, "owed": 7}}):
        baseline = encode_observation(native_game(copy.deepcopy(phase)).observation(0), components)
        public = json.loads(native_game(copy.deepcopy(phase)).observation(0).public_json())
        candidates = encode_candidates([case_move(k) for k in move_cases()], 0, public, components)
        for rotation in range(4):
            obs = native_game(copy.deepcopy(phase), rotation).observation(rotation)
            assert_same(baseline, encode_observation(obs, components))
            assert_same(candidates, encode_candidates([case_move(k) for k in move_cases()], rotation,
                                                      json.loads(obs.public_json()), components))


def test_hand_permutation_invariant(components):
    obs = native_game().observation(0)
    data = json.loads(obs.to_json())
    data["my_hand"].reverse()
    changed = brass.Observation.from_json(json.dumps(data))
    assert_same(encode_observation(obs, components), encode_observation(changed, components))


def test_candidate_permutation_equivariant(components):
    obs = native_game().observation(0)
    moves = json.loads(obs.to_json())["legal_moves"]
    public = json.loads(obs.public_json())
    a = encode_candidates(moves, 0, public, components)
    b = encode_candidates(moves[::-1], 0, public, components)
    for field in fields(a):
        np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name)[::-1])
    assert len(a.features) == obs.num_legal()


@pytest.mark.parametrize("kind", list(move_cases()))
def test_every_move_variant(kind, components):
    public = json.loads(native_game().observation(0).public_json())
    out = encode_candidates([case_move(kind)], 0, public, components)
    schema = components.schema
    assert out.features.dtype == np.float32
    assert out.features[0, schema.fields["candidate"].index("kind:" + kind)] == 1
    assert out.cards[0].sum() == (3 if kind == "Scout" else 1 if "card" in (move_cases()[kind] or {}) else 0)
    assert out.attribute_mask.dtype == np.bool_
    assert out.slot_ids.dtype == np.int64
    args = move_cases()[kind] or {}
    for name in ("industry", "slot", "link", "merchant"):
        col = schema.vocabularies["attributes"].index(name)
        assert out.attribute_mask[0, col] == (name in args)
        if name not in args:
            assert getattr(out, name + "_ids")[0] == -1
    if "link" in args:
        assert out.link_locations[0].tolist() == [14, 15, 21]
        assert out.link_location_mask[0].all()
    if "slot" in args:
        assert out.location_ids[0] == 0
        assert out.target_owner_ids[0] == 0
    for resource, channel, source in (("coal", 0, 1), ("iron", 1, 5)):
        if resource in args:
            plan = args[resource]
            assert out.slot_sources[0, source, channel] == plan["from_tiles"][0][1]
            assert out.source_owners[0, source] == public["slots"][source]["owner"]
            assert out.source_locations[0, source] == components.slot_locations[source]
            assert out.market_quantities[0, channel] == plan["from_market"]
            assert out.market_mask[0, channel]
    if kind in ("Sell", "SellAnother"):
        assert out.slot_sources[0, 3, 2] == 2
        assert out.merchant_sources[0, 0] == 1
    if kind == "SecondRail":
        assert out.slot_sources[0, 3, 2] == 1
    if kind == "Scout":
        assert sorted(out.cards[0][out.cards[0] > 0]) == [1, 2]
    assert out.features[0, schema.fields["candidate"].index("end_action")] == (kind == "EndAction")


@pytest.mark.parametrize("phase", ["ChooseAction", "SecondRail", "DevelopMore", "SellMore",
                                   "BonusDevelop", {"Shortfall": {"player": 0, "owed": 7}}, "GameOver"])
def test_every_phase_native(phase, components):
    game = native_game(copy.deepcopy(phase))
    obs = game.observation(0)
    brass.Observation.from_json(obs.to_json())  # Fixtures obey the strict native transport too.
    out = encode_observation(obs, components)
    name = "Shortfall" if isinstance(phase, dict) else phase
    keys = components.schema.fields["global"]
    assert out.global_features[keys.index("phase:" + name)] == 1
    assert out.global_features[keys.index("shortfall_owed")] == (np.float32(np.log1p(7)) if name == "Shortfall" else 0)
    assert out.candidates.features.shape[0] == obs.num_legal()
    assert out.players.shape == (4, components.schema.dimensions["player"])
    for field in ("global_features", "players", "locations", "slots", "links", "merchants", "hand"):
        assert getattr(out, field).dtype == np.float32
        assert np.isfinite(getattr(out, field)).all()
    if name == "GameOver":
        assert out.candidates.features.shape == (0, components.schema.dimensions["candidate"])


def test_sources_ids_owners_quantities_and_targets_survive(components):
    public = json.loads(native_game().observation(0).public_json())
    base = case_move("Build")
    changes = []
    for path, value in ((["Build", "coal", "from_tiles"], [[1, 3]]),
                        (["Build", "coal", "from_market"], 2),
                        (["Build", "slot"], 4),
                        (["Build", "coal", "from_tiles"], [[5, 2]])):
        variant = copy.deepcopy(base)
        node = variant
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        changes.append(variant)
    out = encode_candidates([base] + changes, 0, public, components)
    for i in range(1, len(changes) + 1):
        assert any(not np.array_equal(getattr(out, f.name)[0], getattr(out, f.name)[i]) for f in fields(out))
    other = copy.deepcopy(public)
    other["slots"][1]["owner"] = 3
    changed = encode_candidates([base], 0, other, components)
    assert out.source_owners[0, 1] != changed.source_owners[0, 1]
    assert not out.source_mask[0, 2]
    assert out.source_owners[0, 2] == -1


def test_public_fields_wilds_and_unclipped_money(components):
    game = wild_public_game()
    for seat in range(4):
        brass.Observation.from_json(game.observation(seat).to_json())
    assert len(json.loads(game.observation(1).to_json())["my_hand"]) <= 8
    out = encode_observation(game.observation(0), components)
    names = components.schema.fields["player"]
    assert out.players[1, names.index("wild_location_in_hand")] == .25
    assert out.players[1, names.index("wild_industry_in_hand")] == .25
    assert out.players[0, names.index("money")] == np.float32(np.log1p(65535))
    assert out.players[1, names.index("has_face_down_discard")] == 1


def test_public_event_uses_no_private_features(components):
    game = native_game()
    previous = game.observation(0).public_json()
    move = next(m for m in game.legal_moves() if "Loan" in m)
    actor = game.current_player()
    game.apply(move)
    event = PublicEvent(1, actor, move, game.observation(0).public_json())
    out = encode_public(event, 0, previous, components)
    # Changing private ordinary own-hand cards while retaining all public information.
    twin = swap_private_hand_with_deck(game)
    other = replace(event, public_json=twin.observation(0).public_json())
    assert_same(out, encode_public(other, 0, previous, components))
    assert out.record == event
    assert out.actor == 0 and out.ply == 1
    assert not hasattr(out.state, "hand")
    assert not hasattr(out.delta, "hand")
    money = components.schema.fields["player"].index("money")
    assert out.delta.players[0, money] > 0
    assert out.move.cards.sum() == 1
    for viewer in range(4):
        rotated = encode_public(event, viewer, previous, components)
        assert rotated.actor == (actor - viewer) % 4
    with pytest.raises(ValueError):
        encode_public(event, 0, game.to_json(), components)
    with pytest.raises(ValueError):
        encode_public(replace(event, public_json=game.observation(0).to_json()), 0, previous, components)


def test_absent_target_owner_has_explicit_mask(components):
    public = json.loads(native_game().observation(0).public_json())
    out = encode_candidates([case_move("Build"), {"Build": dict(move_cases()["Build"], slot=4)},
                             case_move("Loan")], 0, public, components)
    assert out.target_owner_mask.tolist() == [True, False, False]
    assert out.target_owner_ids.tolist() == [0, -1, -1]
    assert out.source_owner_mask[0, 1]
    assert not out.source_owner_mask[0, 2]


def test_public_event_source_owner_uses_pre_move_board(components):
    before = json.loads(native_game().observation(0).public_json())
    after = copy.deepcopy(before)
    after["slots"][1] = None  # Even removal/overbuild must retain the original source identity.
    move = case_move("Rail")
    event = PublicEvent(4, 0, json.dumps(move), json.dumps(after))
    out = encode_public(event, 2, json.dumps(before), components)
    assert out.move.source_owners[0, 1] == 3
    assert out.move.source_locations[0, 1] == 0
    assert out.state.slots[1, 0] == 0
    assert out.delta.slots[1, 0] == -1


@pytest.mark.parametrize("path,value,field", [
    (("round",), 3, "global_features"), (("actions_left",), 2, "global_features"),
    (("turn_idx",), 1, "global_features"), (("coal_market",), 12, "global_features"),
    (("iron_market",), 7, "global_features"), (("deck_size",), 27, "global_features"),
    (("wild_location_supply",), 3, "global_features"),
    (("wild_industry_supply",), 3, "global_features"),
    (("players", 2, "hand_size"), 7, "players"),
    (("players", 2, "wild_location_in_hand"), 1, "players"),
    (("players", 2, "wild_industry_in_hand"), 1, "players"),
    (("players", 2, "has_face_down_discard"), False, "players"),
    (("players", 2, "discard"), [{"location": 1}, {"location": 1}], "players"),
    (("players", 2, "mat_remaining", 0, 0), 2, "players"),
    (("players", 2, "money"), 65534, "players"),
    (("players", 2, "income_space"), 50, "players"),
    (("players", 2, "vp"), 200, "players"), (("players", 2, "spent"), 1000, "players"),
    (("slots", 0, "owner"), 3, "slots"), (("slots", 0, "industry"), "manufacturer", "slots"),
    (("slots", 0, "level"), 3, "slots"), (("slots", 0, "flipped"), True, "slots"),
    (("slots", 1, "resources"), 2, "slots"), (("links", 0), 3, "links"),
    (("merchant_tiles", 0), "blank", "merchants"), (("merchant_beer", 0), False, "merchants"),
])
def test_public_field_changes_survive(path, value, field, components):
    before = json.loads(native_game().observation(0).public_json())
    after = copy.deepcopy(before)
    node = after
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    event = PublicEvent(1, 0, '"EndAction"', json.dumps(after))
    out = encode_public(event, 0, json.dumps(before), components)
    assert np.any(getattr(out.delta, field) != 0)


def test_event_rotated_equivalent_views(components):
    def event_for(rotation):
        game = native_game(rotation=rotation)
        before = game.observation(rotation).public_json()
        move = next(m for m in game.legal_moves() if "Loan" in m)
        game.apply(move)
        event = PublicEvent(1, rotation, move, game.observation(rotation).public_json())
        return encode_public(event, rotation, before, components)
    base = event_for(0)
    for rotation in range(4):
        other = event_for(rotation)
        for name in ("actor", "ply", "state", "delta", "move"):
            assert_same(getattr(base, name), getattr(other, name))


@pytest.mark.parametrize("private_path", [("players", 0, "hand"), ("slots", 0, "rng_seed")])
def test_public_event_rejects_nested_private_fields(private_path, components):
    public = json.loads(native_game().observation(0).public_json())
    before = json.dumps(public)
    node = public
    for key in private_path[:-1]:
        node = node[key]
    node[private_path[-1]] = []
    event = PublicEvent(1, 0, '"EndAction"', json.dumps(public))
    with pytest.raises(ValueError):
        encode_public(event, 0, before, components)


def test_every_variant_occurs_in_native_observation_candidates(components):
    observed = set()
    phases = ("ChooseAction", "SecondRail", "DevelopMore", "SellMore", "BonusDevelop",
              {"Shortfall": {"player": 0, "owed": 7}})
    for phase in phases:
        for era in ("canal", "rail"):
            obs = native_game(copy.deepcopy(phase), era=era).observation(0)
            brass.Observation.from_json(obs.to_json())
            moves = json.loads(obs.to_json())["legal_moves"]
            encoded = encode_observation(obs, components).candidates
            for row, move in enumerate(moves):
                kind = move if isinstance(move, str) else next(iter(move))
                observed.add(kind)
                assert encoded.features[row, components.schema.vocabularies["moves"].index(kind)] == 1
    assert observed == set(move_cases())


def test_schema_records_each_array_shape_dtype(components):
    schema = components.schema
    obs = encode_observation(native_game().observation(0), components)
    n = len(obs.candidates.features)
    for obj, prefix in ((obs, "observation"), (obs.candidates, "candidate")):
        for field in fields(obj):
            value = getattr(obj, field.name)
            if isinstance(value, np.ndarray):
                key = prefix + "." + field.name
                shape = tuple(n if dim == "N" else dim for dim in schema.shapes[key])
                assert value.shape == shape
                assert str(value.dtype) == schema.dtypes[key]
    modified = dict(schema.shapes, **{"candidate.cards": ("N", 99)})
    assert replace(schema, shapes=modified).fingerprint != schema.fingerprint


def test_hand_counts_are_exact_and_source_masks_are_presence(components):
    obs = native_game().observation(0)
    out = encode_observation(obs, components)
    cards = json.loads(obs.to_json())["my_hand"]
    vocab = components.schema.vocabularies["cards"]
    assert out.hand.sum() == len(cards)
    from brass_rl.schema import card_key
    for card in cards:
        assert out.hand[vocab.index(card_key(card))] == cards.count(card)
    np.testing.assert_array_equal(out.candidates.source_mask, out.candidates.slot_sources.any(axis=2))
    np.testing.assert_array_equal(out.candidates.market_mask, out.candidates.market_quantities > 0)
    np.testing.assert_array_equal(out.candidates.merchant_source_mask, out.candidates.merchant_sources > 0)


def test_shortfall_actor_is_independent_of_turn_track(components):
    obs = native_game({"Shortfall": {"player": 2, "owed": 3}}).observation(2)
    out = encode_observation(obs, components)
    names = components.schema.fields["global"]
    assert out.global_features[names.index("actor:0")] == 1
    assert out.global_features[names.index("shortfall_player:0")] == 1
    assert out.global_features[names.index("turn_order:0:2")] == 1
    assert out.global_features[names.index("shortfall_owed")] == np.float32(np.log1p(3))


def test_terminal_and_nonacting_observation_shapes(components):
    for obs in (native_game().observation(1), native_game("GameOver").observation(0)):
        out = encode_observation(obs, components)
        assert obs.num_legal() == 0
        assert out.candidates.cards.shape == (0, 27)
        assert out.candidates.slot_sources.shape == (0, 49, 3)
        assert out.candidates.source_mask.shape == (0, 49)
    terminal = encode_observation(native_game("GameOver").observation(0), components)
    names = components.schema.fields["global"]
    assert terminal.global_features[names.index("game_over")] == 1
    assert sum(terminal.global_features[names.index("actor:%d" % p)] for p in range(4)) == 0


@pytest.mark.parametrize("move", [
    {"Loan": {"card": {"location": 1}, "private_hand": ["SECRET_SENTINEL"]}},
    {"Loan": {"card": {"location": 1}}, "private_hand": ["SECRET_SENTINEL"]},
    {"Build": dict(move_cases()["Build"], coal={"from_tiles": [[1, 1]], "from_market": 0,
                                               "private_hand": ["SECRET_SENTINEL"]})},
    {"Build": dict(move_cases()["Build"], iron={"from_tiles": [[5, 1, "SECRET_SENTINEL"]],
                                               "from_market": 0})},
    {"Loan": {"card": {"location": 1, "private_hand": ["SECRET_SENTINEL"]}}},
    {"Sell": dict(move_cases()["Sell"], beer=[{"Brewery": 3, "private_hand": ["SECRET_SENTINEL"]}])},
    {"Sell": dict(move_cases()["Sell"], beer=[{"Merchant": 0, "Brewery": 3}])},
    {"Scout": {"cards": [{"location": 1}, {"location": 1}, {"private_hand": ["SECRET_SENTINEL"]}]}},
    {"EndAction": {}},
    "Loan",
])
def test_public_event_rejects_private_or_malformed_move_fields(move, components):
    public = native_game().observation(0).public_json()
    event = PublicEvent(1, 0, json.dumps(move), public)
    with pytest.raises(ValueError):
        encode_public(event, 0, public, components)


@pytest.mark.parametrize("kind", list(move_cases()))
def test_public_event_validates_and_preserves_every_move(kind, components):
    public = native_game().observation(0).public_json()
    event = PublicEvent(1, 0, json.dumps(case_move(kind)), public)
    encoded = encode_public(event, 0, public, components)
    assert encoded.record is event
    assert encoded.move.features[0, components.schema.vocabularies["moves"].index(kind)] == 1


@pytest.mark.parametrize("move_json", [
    '{"Loan":{"card":{"location":"SECRET_SENTINEL"},"card":{"location":1}}}',
    '{"Build":{"card":{"location":1},"industry":"cotton","slot":0,"coal":'
    '{"from_tiles":[],"from_market":NaN},"iron":{"from_tiles":[],"from_market":0}}}',
])
def test_public_event_rejects_ambiguous_move_json(move_json, components):
    public = native_game().observation(0).public_json()
    with pytest.raises(ValueError):
        encode_public(PublicEvent(1, 0, move_json, public), 0, public, components)


@pytest.mark.parametrize("in_baseline", [False, True])
def test_public_event_rejects_duplicate_snapshot_fields(in_baseline, components):
    public = native_game().observation(0).public_json()
    ambiguous = public.replace('"discard":[]',
                               '"discard":[{"location":"SECRET_SENTINEL"}],"discard":[]', 1)
    assert ambiguous != public
    event = PublicEvent(1, 0, '"EndAction"', public if in_baseline else ambiguous)
    with pytest.raises(ValueError):
        encode_public(event, 0, ambiguous if in_baseline else public, components)
