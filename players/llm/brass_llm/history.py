"""Wire allowlists for native PublicEvent hooks. The native referee owns legality."""

import json
from dataclasses import dataclass

from .protocol import LLMError


@dataclass(frozen=True)
class Array:
    item: object
    length: int = -1


def _matches(value, schema):
    if isinstance(schema, dict):
        return (
            type(value) is dict
            and value.keys() == schema.keys()
            and all(_matches(value[k], spec) for k, spec in schema.items())
        )
    if isinstance(schema, Array):
        return (
            type(value) is list
            and (
                len(value) == schema.length if schema.length >= 0 else len(value) <= 128
            )
            and all(_matches(v, schema.item) for v in value)
        )
    if isinstance(schema, tuple):
        return any(_matches(value, option) for option in schema)
    if isinstance(schema, set):
        return type(value) is str and value in schema
    if isinstance(schema, range):
        return type(value) is int and value in schema
    return type(value) is schema


BYTE = range(256)
MONEY = range(65536)
SEAT = range(4)
SLOT = range(49)
LINK = range(39)
MERCHANT = range(9)
INDUSTRY = {"cotton", "manufacturer", "pottery", "coal", "iron", "brewery"}
CARD = (
    {"location": range(20)},
    {"industry": {"iron", "coal", "pottery", "brewery", "cotton-or-manufacturer"}},
    {"wild-location", "wild-industry"},
)
PLAN = {"from_market": BYTE, "from_tiles": Array(Array(BYTE, 2))}
BEER = Array(({"Brewery": SLOT}, {"Merchant": MERCHANT}))
TILE = {
    "owner": SEAT,
    "industry": INDUSTRY,
    "level": BYTE,
    "flipped": bool,
    "resources": BYTE,
}
PLAYER = {
    "hand_size": BYTE,
    "wild_location_in_hand": BYTE,
    "wild_industry_in_hand": BYTE,
    "discard": Array(CARD),
    "has_face_down_discard": bool,
    "mat_remaining": Array(Array(BYTE, 8), 6),
    "money": MONEY,
    "income_space": BYTE,
    "vp": MONEY,
    "spent": MONEY,
}
PHASE = (
    {
        "ChooseAction",
        "SecondRail",
        "DevelopMore",
        "SellMore",
        "BonusDevelop",
        "GameOver",
    },
    {"Shortfall": {"player": SEAT, "owed": MONEY}},
)
PUBLIC = {
    "era": {"rail", "canal"},
    "round": BYTE,
    "turn_order": Array(SEAT, 4),
    "turn_idx": SEAT,
    "actions_left": BYTE,
    "phase": PHASE,
    "slots": Array((type(None), TILE), 49),
    "links": Array((type(None), SEAT), 39),
    "merchant_tiles": Array({"any", "cotton", "manufacturer", "pottery", "blank"}, 9),
    "merchant_beer": Array(bool, 9),
    "coal_market": BYTE,
    "iron_market": BYTE,
    "deck_size": BYTE,
    "wild_location_supply": BYTE,
    "wild_industry_supply": BYTE,
    "players": Array(PLAYER, 4),
    "game_over": bool,
}
BUILD = {"card": CARD, "industry": INDUSTRY, "slot": SLOT, "coal": PLAN, "iron": PLAN}
DEVELOP = {"industry": INDUSTRY, "iron": PLAN}
SELL = {"slot": SLOT, "merchant": MERCHANT, "beer": BEER}
ACTION = (
    {"Build": BUILD},
    {"Canal": {"card": CARD, "link": LINK}},
    {"Rail": {"card": CARD, "link": LINK, "coal": PLAN}},
    {"Develop": dict(DEVELOP, card=CARD)},
    {"Sell": dict(SELL, card=CARD)},
    {"Loan": {"card": CARD}},
    {"Pass": {"card": CARD}},
    {"Scout": {"cards": Array(CARD, 3)}},
)
MOVES = {
    "ChooseAction": ACTION,
    "SecondRail": (
        {"SecondRail": {"link": LINK, "coal": PLAN, "beer": SLOT}},
        {"EndAction"},
    ),
    "DevelopMore": ({"DevelopSecond": DEVELOP}, {"EndAction"}),
    "SellMore": ({"SellAnother": SELL}, {"EndAction"}),
    "BonusDevelop": {"BonusDevelop": {"industry": INDUSTRY}},
    "Shortfall": {"ShortfallRemove": {"slot": SLOT}},
}


def checked_json(raw, schema):
    try:
        value = json.loads(raw)
        if _matches(value, schema):
            return value
    except (ValueError, TypeError, RecursionError):
        pass
    raise LLMError("history")


def public_snapshot(raw):
    return checked_json(raw, PUBLIC)


def public_move(raw, phase):
    return checked_json(raw, MOVES["Shortfall" if isinstance(phase, dict) else phase])
