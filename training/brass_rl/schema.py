"""Schema v1: canonical engine components, vocabularies and numeric field contracts."""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Tuple

import brass
import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def card_key(card: Any) -> str:
    """Stable vocabulary key, independent of dictionary insertion or hand order."""
    return canonical_json(card)


MOVE_KINDS = ("Build", "Canal", "Rail", "Develop", "Sell", "Loan", "Scout", "Pass",
              "SecondRail", "DevelopSecond", "SellAnother", "BonusDevelop",
              "ShortfallRemove", "EndAction")
PHASES = ("ChooseAction", "SecondRail", "DevelopMore", "SellMore", "BonusDevelop",
          "Shortfall", "GameOver")
ATTRIBUTES = ("industry", "slot", "location", "link", "merchant")


@dataclass(frozen=True)
class Schema:
    version: int
    component_hash: str
    vocabularies: Mapping[str, Tuple[str, ...]]
    dimensions: Mapping[str, int]
    scales: Mapping[str, str]
    fields: Mapping[str, Tuple[str, ...]]
    shapes: Mapping[str, Tuple[Any, ...]]
    dtypes: Mapping[str, str]

    def canonical_json(self) -> str:
        return canonical_json(asdict(self))

    @property
    def fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComponentData:
    data: Dict[str, Any]
    component_hash: str
    schema: Schema
    slot_locations: np.ndarray
    link_locations: np.ndarray
    link_location_mask: np.ndarray
    merchant_locations: np.ndarray

    @classmethod
    def load(cls) -> "ComponentData":
        data = json.loads(brass.component_data())
        if data["version"] != 1:
            raise ValueError("unsupported engine component version")
        component_hash = digest(data)
        cards = tuple(sorted({card_key(c) for c in data["deck_4p"]} |
                             {card_key("wild-location"), card_key("wild-industry")}))
        industries = tuple(data["industries"])
        vocab = {"cards": cards, "industries": industries, "moves": MOVE_KINDS,
                 "phases": PHASES, "attributes": ATTRIBUTES,
                 "location_kinds": tuple(sorted({l["kind"] for l in data["locations"]})),
                 "merchant_types": tuple(sorted(set(data["merchant_tiles_4p"])))}
        fields = {
            "global": tuple(["era:canal", "era:rail"] + ["phase:" + p for p in PHASES] +
                            ["round", "actions_left", "turn_idx", "coal_market", "iron_market",
                             "deck_size", "wild_location_supply", "wild_industry_supply", "game_over"] +
                            ["turn_order:%d:%d" % (i, p) for i in range(4) for p in range(4)] +
                            ["actor:%d" % p for p in range(4)] +
                            ["shortfall_player:%d" % p for p in range(4)] +
                            ["shortfall_present", "shortfall_owed"]),
            "player": tuple(["hand_size", "wild_location_in_hand", "wild_industry_in_hand",
                             "has_face_down_discard", "money", "income_space", "income_level", "vp", "spent"] +
                            ["mat:%s:%d" % (i, level) for i in industries for level in range(1, 9)] +
                            ["discard:" + c for c in cards]),
            "location": tuple("kind:" + k for k in vocab["location_kinds"]),
            "slot": tuple(["present"] + ["owner:%d" % p for p in range(4)] +
                          ["industry:" + i for i in industries] + ["level", "flipped", "resources"] +
                          ["allows:" + i for i in industries]),
            "link": ("present", "owner:0", "owner:1", "owner:2", "owner:3", "canal", "rail"),
            "merchant": tuple(["type:" + t for t in vocab["merchant_types"]] + ["beer"]),
            "candidate": tuple(["kind:" + k for k in MOVE_KINDS] +
                               ["end_action", "coal_present", "iron_present", "beer_present"]),
        }
        dimensions = {name: len(value) for name, value in fields.items()}
        dimensions.update(cards=len(cards), players=4, locations=len(data["locations"]),
                          slots=len(data["slots"]), links=len(data["links"]),
                          merchants=len(data["merchant_tiles_4p"]), source_channels=3,
                          attributes=len(ATTRIBUTES), link_incidence=3)
        scales = {"round": "x/10", "actions_left": "x/2", "turn_idx": "x/3",
                  "coal_market": "x/len(coal_market_prices)", "iron_market": "x/len(iron_market_prices)",
                  "deck_size": "x/len(deck_4p)", "wild_location_supply": "x/WILD_LOCATION_COUNT",
                  "wild_industry_supply": "x/WILD_INDUSTRY_COUNT", "hand_size": "x/HAND_SIZE",
                  "wild_location_in_hand": "x/WILD_LOCATION_COUNT",
                  "wild_industry_in_hand": "x/WILD_INDUSTRY_COUNT", "money": "log1p(x)",
                  "spent": "log1p(x)", "shortfall_owed": "log1p(x)",
                  "income_space": "x/MAX_INCOME_SPACE", "income_level": "x/MAX_INCOME_LEVEL",
                  "vp": "x/100", "mat": "x/3", "discard": "x/8", "level": "x/8",
                  "resources": "x/8", "hand": "raw counts", "candidate_cards": "raw counts",
                  "source_quantities": "raw counts", "categorical": "one-hot or int64 ID",
                  "mask": "bool (entity presence flags in float feature rows are 0/1)",
                  "delta": "encoded(after)-encoded(before); no clipping"}
        shapes = {
            "observation.global_features": (dimensions["global"],),
            "observation.players": (4, dimensions["player"]),
            "observation.locations": (dimensions["locations"], dimensions["location"]),
            "observation.slots": (dimensions["slots"], dimensions["slot"]),
            "observation.links": (dimensions["links"], dimensions["link"]),
            "observation.merchants": (dimensions["merchants"], dimensions["merchant"]),
            "observation.hand": (len(cards),),
            "candidate.features": ("N", dimensions["candidate"]),
            "candidate.cards": ("N", len(cards)),
            "candidate.attribute_mask": ("N", len(ATTRIBUTES)),
            "candidate.link_locations": ("N", 3),
            "candidate.link_location_mask": ("N", 3),
            "candidate.slot_sources": ("N", dimensions["slots"], 3),
            "candidate.merchant_sources": ("N", dimensions["merchants"]),
            "candidate.merchant_source_mask": ("N", dimensions["merchants"]),
            "candidate.market_quantities": ("N", 2),
            "candidate.market_mask": ("N", 2),
        }
        for name in ATTRIBUTES:
            shapes["candidate." + name + "_ids"] = ("N",)
        for name in ("target_owner_ids", "target_owner_mask"):
            shapes["candidate." + name] = ("N",)
        for name in ("source_mask", "source_owners", "source_owner_mask", "source_locations"):
            shapes["candidate." + name] = ("N", dimensions["slots"])
        dtypes = {key: "float32" for key in shapes}
        for name in (*[a + "_ids" for a in ATTRIBUTES], "target_owner_ids", "link_locations",
                     "source_owners", "source_locations"):
            dtypes["candidate." + name] = "int64"
        for key in shapes:
            if key.endswith("mask"):
                dtypes[key] = "bool"
        # Event state/delta reuse the entity contract; neither contains a hand.
        for key in tuple(shapes):
            if key.startswith("observation.") and key != "observation.hand":
                for prefix in ("event.state.", "event.delta."):
                    new_key = prefix + key.split(".", 1)[1]
                    shapes[new_key], dtypes[new_key] = shapes[key], dtypes[key]
        scales.update(ply="raw chronological integer", actor="relative int64 categorical ID",
                      entity_ids="raw fixed engine row index; absent -1 with presence mask")
        schema = Schema(1, component_hash, vocab, dimensions, scales, fields, shapes, dtypes)
        slot_locations = np.array([s["loc"] for s in data["slots"]], dtype=np.int64)
        link_locations = np.array([[l["a"], l["b"], -1 if l["extra"] is None else l["extra"]]
                                   for l in data["links"]], dtype=np.int64)
        merchant_locations = np.array([m["loc"] for m in data["merchant_spaces"]
                                       for _ in range(m["tile_slots"])], dtype=np.int64)
        mask = link_locations >= 0
        for array in (slot_locations, link_locations, mask, merchant_locations):
            array.flags.writeable = False
        return cls(data, component_hash, schema, slot_locations, link_locations, mask,
                   merchant_locations)
