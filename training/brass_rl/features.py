"""Raw structured features from native observations and sanitized public events only.

No tensor embeddings, game objects, hidden-state reconstruction or candidate ranking.
Numerical imports are isolated here/schema.py; Stage 1 need never import this module.
"""
from dataclasses import dataclass, fields
import json
from functools import lru_cache
import math
from typing import Any, Mapping, Sequence

import brass
import numpy as np

from brass_ai.events import PublicEvent
from .schema import ATTRIBUTES, ComponentData, MOVE_KINDS, PHASES, card_key


@dataclass(frozen=True)
class EncodedCandidates:
    features: np.ndarray
    cards: np.ndarray
    industry_ids: np.ndarray
    slot_ids: np.ndarray
    location_ids: np.ndarray
    link_ids: np.ndarray
    merchant_ids: np.ndarray
    attribute_mask: np.ndarray
    target_owner_ids: np.ndarray
    target_owner_mask: np.ndarray
    link_locations: np.ndarray
    link_location_mask: np.ndarray
    slot_sources: np.ndarray
    source_mask: np.ndarray
    source_owners: np.ndarray
    source_owner_mask: np.ndarray
    source_locations: np.ndarray
    merchant_sources: np.ndarray
    merchant_source_mask: np.ndarray
    market_quantities: np.ndarray
    market_mask: np.ndarray


@dataclass(frozen=True)
class EncodedPublic:
    global_features: np.ndarray
    players: np.ndarray
    locations: np.ndarray
    slots: np.ndarray
    links: np.ndarray
    merchants: np.ndarray


@dataclass(frozen=True)
class EncodedObservation(EncodedPublic):
    hand: np.ndarray
    candidates: EncodedCandidates


@dataclass(frozen=True)
class EncodedEvent:
    ply: int
    actor: int
    move: EncodedCandidates
    state: EncodedPublic
    delta: EncodedPublic
    record: PublicEvent


def relative_seat(seat: int, viewer: int) -> int:
    return (seat - viewer) % 4


def validate_events(events: Sequence[EncodedEvent]) -> None:
    for index, event in enumerate(events):
        if not isinstance(event, EncodedEvent):
            raise TypeError("memory requires EncodedEvent records")
        if (isinstance(event.actor, (bool, np.bool_)) or not isinstance(event.actor, (int, np.integer))
                or not 0 <= event.actor < 4):
            raise ValueError("memory event actor must be an integral relative seat in 0..3")
        if any(len(getattr(event.move, f.name)) != 1 for f in fields(EncodedCandidates)):
            raise ValueError("memory event move needs exactly one candidate row")
        if index and event.ply != events[index - 1].ply + 1:
            raise ValueError("memory events must be in consecutive ply order")


def _viewer(viewer: int) -> None:
    if isinstance(viewer, bool) or not isinstance(viewer, int) or viewer not in range(4):
        raise ValueError("viewer must be a seat in 0..3")


def _card_token(card):
    # Keep JSON scalar types distinct: True and 1 compare equal in Python maps.
    if type(card) is str:
        return (str, card)
    if type(card) is dict and len(card) == 1:
        key, value = next(iter(card.items()))
        if type(key) is str and type(value) in (int, str):
            return (key, type(value), value)
    return None


@lru_cache(maxsize=16)
def _card_indices(vocab):
    # Only the fixed schema vocabulary enters this process cache.
    return {_card_token(json.loads(key)): index for index, key in enumerate(vocab)}


def _card_index(card, vocab, indices):
    token = _card_token(card)
    if token is not None and token in indices:
        return indices[token]
    return vocab.index(card_key(card))


def _counts(cards: Sequence[Any], components: ComponentData) -> np.ndarray:
    vocab = components.schema.vocabularies["cards"]
    indices = _card_indices(vocab)
    counts = [0] * len(vocab)
    for card in cards:
        counts[_card_index(card, vocab, indices)] += 1
    return np.asarray(counts, dtype=np.float32)


def _public(public: Mapping[str, Any], viewer: int, components: ComponentData) -> EncodedPublic:
    """Shared entity encoding: public event state and current observation agree exactly."""
    schema, data = components.schema, components.data
    dims, vocab = schema.dimensions, schema.vocabularies
    const = data["constants"]
    values = {name: 0.0 for name in schema.fields["global"]}
    values["era:" + public["era"]] = 1
    phase = public["phase"]
    name = "Shortfall" if isinstance(phase, dict) else phase
    values["phase:" + name] = 1
    for name, divisor in (("round", 10), ("actions_left", 2), ("turn_idx", 3),
                          ("coal_market", len(data["coal_market_prices"])),
                          ("iron_market", len(data["iron_market_prices"])),
                          ("deck_size", len(data["deck_4p"])),
                          ("wild_location_supply", const["WILD_LOCATION_COUNT"]),
                          ("wild_industry_supply", const["WILD_INDUSTRY_COUNT"]), ("game_over", 1)):
        values[name] = public[name] / divisor
    for i, p in enumerate(public["turn_order"]):
        values["turn_order:%d:%d" % (i, relative_seat(p, viewer))] = 1
    if isinstance(phase, dict):
        shortfall = phase["Shortfall"]
        actor = shortfall["player"]
        values["shortfall_player:%d" % relative_seat(actor, viewer)] = 1
        values["shortfall_present"] = 1
        values["shortfall_owed"] = math.log1p(shortfall["owed"])
    else:
        actor = public["turn_order"][public["turn_idx"]] if not public["game_over"] else None
    if actor is not None:
        values["actor:%d" % relative_seat(actor, viewer)] = 1
    global_features = np.array([values[n] for n in schema.fields["global"]], dtype=np.float32)
    players = np.zeros((4, dims["player"]), dtype=np.float32)
    for p, player in enumerate(public["players"]):
        row = [player["hand_size"] / const["HAND_SIZE"],
               player["wild_location_in_hand"] / const["WILD_LOCATION_COUNT"],
               player["wild_industry_in_hand"] / const["WILD_INDUSTRY_COUNT"],
               player["has_face_down_discard"], math.log1p(player["money"]),
               player["income_space"] / const["MAX_INCOME_SPACE"],
               data["income_levels"][player["income_space"]] / const["MAX_INCOME_LEVEL"],
               player["vp"] / 100, math.log1p(player["spent"])]
        row.extend(n / 3 for industry in player["mat_remaining"] for n in industry)
        row.extend(_counts(player["discard"], components) / 8)
        players[relative_seat(p, viewer)] = row
    locations = np.zeros((dims["locations"], dims["location"]), dtype=np.float32)
    for i, loc in enumerate(data["locations"]):
        locations[i, vocab["location_kinds"].index(loc["kind"])] = 1
    slots = np.zeros((dims["slots"], dims["slot"]), dtype=np.float32)
    for i, (definition, tile) in enumerate(zip(data["slots"], public["slots"])):
        for industry in definition["industries"]:
            slots[i, 14 + vocab["industries"].index(industry)] = 1
        if tile is not None:
            slots[i, 0] = 1
            slots[i, 1 + relative_seat(tile["owner"], viewer)] = 1
            slots[i, 5 + vocab["industries"].index(tile["industry"])] = 1
            slots[i, 11:14] = [tile["level"] / 8, tile["flipped"], tile["resources"] / 8]
    links = np.zeros((dims["links"], dims["link"]), dtype=np.float32)
    for i, (definition, owner) in enumerate(zip(data["links"], public["links"])):
        links[i, 5:] = [definition["canal"], definition["rail"]]
        if owner is not None:
            links[i, 0] = 1
            links[i, 1 + relative_seat(owner, viewer)] = 1
    merchants = np.zeros((dims["merchants"], dims["merchant"]), dtype=np.float32)
    for i, (tile, beer) in enumerate(zip(public["merchant_tiles"], public["merchant_beer"])):
        merchants[i, vocab["merchant_types"].index(tile)] = 1
        merchants[i, -1] = beer
    return EncodedPublic(global_features, players, locations, slots, links, merchants)


def encode_candidates(moves: Sequence[Any], viewer: int, public: Mapping[str, Any],
                      components: ComponentData) -> EncodedCandidates:
    """Numerical helper for moves already supplied by a trusted observation/event.

    Preserves input row order, supports isolated semantic/permutation tests, and performs
    no legality check. This helper cannot manufacture native observations for policy use.
    """
    _viewer(viewer)
    dims, vocab = components.schema.dimensions, components.schema.vocabularies
    n, s, m = len(moves), dims["slots"], dims["merchants"]
    features = np.zeros((n, dims["candidate"]), dtype=np.float32)
    cards = np.zeros((n, dims["cards"]), dtype=np.float32)
    ids = {name: np.full(n, -1, dtype=np.int64) for name in ATTRIBUTES}
    mask = np.zeros((n, len(ATTRIBUTES)), dtype=np.bool_)
    target_owners = np.full(n, -1, dtype=np.int64)
    incidences = np.full((n, 3), -1, dtype=np.int64)
    incidence_mask = np.zeros((n, 3), dtype=np.bool_)
    sources = np.zeros((n, s, 3), dtype=np.float32)
    merchant_sources = np.zeros((n, m), dtype=np.float32)
    market = np.zeros((n, 2), dtype=np.float32)
    market_mask = np.zeros((n, 2), dtype=np.bool_)
    for row, move in enumerate(moves):
        kind, args = (move, {}) if isinstance(move, str) else next(iter(move.items()))
        features[row, MOVE_KINDS.index(kind)] = 1
        features[row, 14] = kind == "EndAction"
        cards[row] = _counts(args.get("cards", [args["card"]] if "card" in args else []), components)
        for name in ATTRIBUTES:
            if name in args:
                ids[name][row] = vocab["industries"].index(args[name]) if name == "industry" else args[name]
                mask[row, ATTRIBUTES.index(name)] = True
        if "slot" in args:
            slot = args["slot"]
            ids["location"][row] = components.slot_locations[slot]
            mask[row, ATTRIBUTES.index("location")] = True
            tile = public["slots"][slot]
            if tile is not None:
                target_owners[row] = relative_seat(tile["owner"], viewer)
        if "link" in args:
            link = args["link"]
            incidences[row] = components.link_locations[link]
            incidence_mask[row] = components.link_location_mask[link]
            if public["links"][link] is not None:
                target_owners[row] = relative_seat(public["links"][link], viewer)
        for channel, resource in enumerate(("coal", "iron")):
            if resource in args:
                features[row, 15 + channel] = 1
                plan = args[resource]
                for slot, quantity in plan["from_tiles"]:
                    sources[row, slot, channel] += quantity
                market[row, channel] = plan["from_market"]
                market_mask[row, channel] = plan["from_market"] > 0
        if "beer" in args:
            features[row, 17] = 1
            beer = [{"Brewery": args["beer"]}] if kind == "SecondRail" else args["beer"]
            for source in beer:
                if "Brewery" in source:
                    sources[row, source["Brewery"], 2] += 1
                else:
                    merchant_sources[row, source["Merchant"]] += 1
    source_mask = sources.any(axis=2)
    source_owners = np.full((n, s), -1, dtype=np.int64)
    source_locations = np.full((n, s), -1, dtype=np.int64)
    rows, slots = np.nonzero(source_mask)
    source_locations[rows, slots] = components.slot_locations[slots]
    for row, slot in zip(rows, slots):
        tile = public["slots"][slot]
        if tile is not None:
            source_owners[row, slot] = relative_seat(tile["owner"], viewer)
    return EncodedCandidates(features, cards, ids["industry"], ids["slot"], ids["location"],
                             ids["link"], ids["merchant"], mask, target_owners, target_owners >= 0, incidences,
                             incidence_mask, sources, source_mask, source_owners, source_owners >= 0, source_locations,
                             merchant_sources, merchant_sources > 0, market, market_mask)


def encode_observation(obs: brass.Observation, components: ComponentData) -> EncodedObservation:
    if not isinstance(obs, brass.Observation):
        raise TypeError("encode_observation requires a native brass.Observation")
    view = json.loads(obs.to_json())
    state = _public(view["public"], obs.me, components)
    return EncodedObservation(**{f.name: getattr(state, f.name) for f in fields(state)},
                              hand=_counts(view["my_hand"], components),
                              candidates=encode_candidates(view["legal_moves"], obs.me,
                                                           view["public"], components))


_PUBLIC_KEYS = frozenset(("era", "round", "turn_order", "turn_idx", "actions_left", "phase",
                          "slots", "links", "merchant_tiles", "merchant_beer", "coal_market",
                          "iron_market", "deck_size", "wild_location_supply", "wild_industry_supply",
                          "players", "game_over"))
_PLAYER_KEYS = frozenset(("hand_size", "wild_location_in_hand", "wild_industry_in_hand", "discard",
                          "has_face_down_discard", "mat_remaining", "money", "income_space", "vp", "spent"))
_TILE_KEYS = frozenset(("owner", "industry", "level", "flipped", "resources"))


def _keys(value: Any, expected: frozenset) -> None:
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError("expected exact sanitized PublicSnapshot fields")


def _public_json(text: str, components: ComponentData) -> Mapping[str, Any]:
    public = _record_json(text)
    _keys(public, _PUBLIC_KEYS)
    dims = components.schema.dimensions
    for key, count in (("players", 4), ("turn_order", 4), ("slots", dims["slots"]),
                       ("links", dims["links"]), ("merchant_tiles", dims["merchants"]),
                       ("merchant_beer", dims["merchants"])):
        if not isinstance(public[key], list) or len(public[key]) != count:
            raise ValueError("invalid PublicSnapshot " + key)
    for player in public["players"]:
        _keys(player, _PLAYER_KEYS)
        vocab = components.schema.vocabularies["cards"]
        indices = _card_indices(vocab)
        for card in player["discard"]:
            _card_index(card, vocab, indices)
        mat = player["mat_remaining"]
        if len(mat) != 6 or any(len(row) != 8 for row in mat):
            raise ValueError("invalid PublicSnapshot mat")
    for tile in public["slots"]:
        if tile is not None:
            _keys(tile, _TILE_KEYS)
    phase = public["phase"]
    if isinstance(phase, dict):
        _keys(phase, frozenset(("Shortfall",)))
        _keys(phase["Shortfall"], frozenset(("player", "owed")))
    elif phase not in PHASES:
        raise ValueError("invalid PublicSnapshot phase")
    return public


_MOVE_FIELDS = {
    "Build": frozenset(("card", "industry", "slot", "coal", "iron")),
    "Canal": frozenset(("card", "link")),
    "Rail": frozenset(("card", "link", "coal")),
    "Develop": frozenset(("card", "industry", "iron")),
    "Sell": frozenset(("card", "slot", "merchant", "beer")),
    "Loan": frozenset(("card",)),
    "Scout": frozenset(("cards",)),
    "Pass": frozenset(("card",)),
    "SecondRail": frozenset(("link", "coal", "beer")),
    "DevelopSecond": frozenset(("industry", "iron")),
    "SellAnother": frozenset(("slot", "merchant", "beer")),
    "BonusDevelop": frozenset(("industry",)),
    "ShortfallRemove": frozenset(("slot",)),
}


def _unique_object(pairs: Sequence[Any]) -> Mapping[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate public event record key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite public event record value: " + value)


def _record_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _move_uint(value: Any, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid public event move integer")


def _event_move(text: str, components: ComponentData) -> Any:
    """Validate the full retained record, not just attributes used numerically.

    This is structural sanitation, independent of engine legality. Reject duplicate
    keys too: parsing then retaining the original string must not hide extra data.
    """
    move = _record_json(text)
    if move == "EndAction":
        return move
    if not isinstance(move, dict) or len(move) != 1:
        raise ValueError("expected one public event move variant")
    kind, args = next(iter(move.items()))
    if kind not in _MOVE_FIELDS:
        raise ValueError("unknown public event move variant")
    _keys(args, _MOVE_FIELDS[kind])
    dims, vocab = components.schema.dimensions, components.schema.vocabularies
    cards = []
    if "card" in args:
        cards.append(args["card"])
    if "cards" in args:
        if not isinstance(args["cards"], list) or len(args["cards"]) != 3:
            raise ValueError("Scout must contain three public cards")
        cards.extend(args["cards"])
    for card in cards:
        # The finite vocabulary consists of exact native card structures only.
        if card_key(card) not in vocab["cards"]:
            raise ValueError("invalid public event move card")
    if "industry" in args and args["industry"] not in vocab["industries"]:
        raise ValueError("invalid public event move industry")
    for field, dimension in (("slot", "slots"), ("link", "links"), ("merchant", "merchants")):
        if field in args:
            _move_uint(args[field], dims[dimension] - 1)
    for resource in ("coal", "iron"):
        if resource not in args:
            continue
        plan = args[resource]
        _keys(plan, frozenset(("from_tiles", "from_market")))
        _move_uint(plan["from_market"], 255)
        if not isinstance(plan["from_tiles"], list):
            raise ValueError("invalid public event resource sources")
        for source in plan["from_tiles"]:
            if not isinstance(source, list) or len(source) != 2:
                raise ValueError("expected public resource slot/count pair")
            _move_uint(source[0], dims["slots"] - 1)
            _move_uint(source[1], 255, 1)
    if "beer" in args:
        if kind == "SecondRail":
            _move_uint(args["beer"], dims["slots"] - 1)
        else:
            if not isinstance(args["beer"], list):
                raise ValueError("invalid public event beer sources")
            for source in args["beer"]:
                if not isinstance(source, dict) or len(source) != 1:
                    raise ValueError("expected one public beer source")
                source_kind, entity = next(iter(source.items()))
                if source_kind not in ("Brewery", "Merchant"):
                    raise ValueError("unknown public beer source")
                dimension = "slots" if source_kind == "Brewery" else "merchants"
                _move_uint(entity, dims[dimension] - 1)
    return move


def encode_public(event: PublicEvent, viewer: int, previous_public_json: str,
                  components: ComponentData) -> EncodedEvent:
    """Encode one public transition; move source metadata uses the pre-move board."""
    encoded, _ = _encode_transition(event, viewer, previous_public_json, components)
    return encoded


def _encode_transition(event, viewer, previous_json, components, before=None, prior=None):
    """Consume a previously validated snapshot owned by this viewer's lifecycle."""
    if not isinstance(event, PublicEvent):
        raise TypeError("encode_public requires a PublicEvent")
    _viewer(viewer)
    if isinstance(event.ply, bool) or not isinstance(event.ply, int) or event.ply < 1:
        raise ValueError("event ply must start at one")
    _viewer(event.actor)
    validated_move = _event_move(event.move_json, components)
    if before is None:
        before = _public_json(previous_json, components)
    after = _public_json(event.public_json, components)
    state = _public(after, viewer, components)
    if prior is None:
        prior = _public(before, viewer, components)
    delta = EncodedPublic(**{f.name: getattr(state, f.name) - getattr(prior, f.name) for f in fields(state)})
    move = encode_candidates([validated_move], viewer, before, components)
    return EncodedEvent(event.ply, relative_seat(event.actor, viewer), move, state, delta, event), after
