# SPDX-License-Identifier: Apache-2.0
"""Optional-dependency subprocesses and explicit privileged native test fixtures."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_without_torch(code):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "engine/tools"), str(ROOT / "training")]))
    blocker = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ModuleNotFoundError('torch deliberately blocked')
sys.meta_path.insert(0, NoTorch())
"""
    return subprocess.run([sys.executable, "-c", blocker + code], env=env,
                          cwd=ROOT, capture_output=True, text=True, timeout=30)


def native_game(phase="ChooseAction", rotation=0, era="canal"):
    """Conservation-valid fixture; privileged construction stays in tests only."""
    import brass
    import json
    d = json.loads(brass.Game(123).to_json())
    d["turn_order"] = [0, 3, 1, 2]
    d["phase"] = phase
    d["era"] = era
    if phase in ("SecondRail", "GameOver"):
        d["era"] = "rail"
    d["game_over"] = phase == "GameOver"
    # Public stats and stocks differ by seat, to expose incomplete seat rotations.
    for p in range(4):
        d["players"][p]["money"] = 100 + p
        d["players"][p]["vp"] = 10 + p
        d["players"][p]["spent"] = p
    for slot, owner, industry, level, resources in (
        (0, 0, "cotton", 2, 0), (1, 1, "coal", 2, 3),
        (5, 2, "iron", 2, 4), (3, 3, "brewery", 2, 2),
        (13, 0, "brewery", 2, 2),
    ):
        if industry == "brewery" and d["era"] == "canal":
            resources = 1
        d["slots"][slot] = dict(owner=owner, industry=industry, level=level,
                                resources=resources, flipped=False)
        industry_id = ["cotton", "manufacturer", "pottery", "coal", "iron", "brewery"].index(industry)
        d["players"][owner]["mat_remaining"][industry_id][level - 1] -= 1
    d["links"][0] = 0
    d["links"][23] = 1
    # Reach a cotton buyer in Nottingham for actual Sell/SellAnother candidates.
    d["merchant_tiles"][3], d["merchant_tiles"][7] = d["merchant_tiles"][7], d["merchant_tiles"][3]
    d["merchant_beer"][3], d["merchant_beer"][7] = d["merchant_beer"][7], d["merchant_beer"][3]
    if phase == "GameOver":
        for player in d["players"]:
            player["discard"].extend(player["hand"])
            player["hand"] = []
        d["players"][0]["discard"].extend(d["deck"])
        d["deck"] = []
    if isinstance(phase, dict):
        d["players"][phase["Shortfall"]["player"]]["money"] = 0
    # Relabel physical seats without changing geography or the viewer's information.
    if rotation:
        players = d["players"][:]
        for p in range(4):
            d["players"][(p + rotation) % 4] = players[p]
        d["turn_order"] = [(p + rotation) % 4 for p in d["turn_order"]]
        d["links"] = [None if p is None else (p + rotation) % 4 for p in d["links"]]
        for tile in d["slots"]:
            if tile:
                tile["owner"] = (tile["owner"] + rotation) % 4
        if isinstance(d["phase"], dict):
            d["phase"]["Shortfall"]["player"] = (d["phase"]["Shortfall"]["player"] + rotation) % 4
    return brass.Game.from_json(json.dumps(d))


def move_cases():
    """Explicit semantic cases, independent of engine canonical candidate ordering."""
    card = {"location": 1}
    coal = {"from_tiles": [[1, 2]], "from_market": 1}
    iron = {"from_tiles": [[5, 1]], "from_market": 2}
    beer = [{"Brewery": 3}, {"Brewery": 3}, {"Merchant": 0}]
    return {
        "Build": dict(card=card, industry="cotton", slot=0, coal=coal, iron=iron),
        "Canal": dict(card=card, link=29),
        "Rail": dict(card=card, link=29, coal=coal),
        "Develop": dict(card=card, industry="iron", iron=iron),
        "Sell": dict(card=card, slot=0, merchant=0, beer=beer),
        "Loan": dict(card=card),
        "Scout": dict(cards=[card, card, {"industry": "iron"}]),
        "Pass": dict(card=card),
        "SecondRail": dict(link=29, coal=coal, beer=3),
        "DevelopSecond": dict(industry="iron", iron=iron),
        "SellAnother": dict(slot=0, merchant=0, beer=beer),
        "BonusDevelop": dict(industry="iron"),
        "ShortfallRemove": dict(slot=0),
        "EndAction": None,
    }


def case_move(kind):
    case = move_cases()[kind]
    return kind if case is None else {kind: case}


def shortfall_game():
    """Two-choice synthetic optimizer fixture; not complete gameplay evidence."""
    import brass
    import json
    state = json.loads(native_game({'Shortfall': {'player': 0, 'owed': 100}}).to_json())
    state['slots'][5]['owner'] = 0
    state['players'][2]['mat_remaining'][4][1] += 1
    state['players'][0]['mat_remaining'][4][1] -= 1
    return brass.Game.from_json(json.dumps(state))


def wild_public_game():
    """Obtain both wilds by a legal Scout; retain native hand/card conservation."""
    import brass
    import json
    game = native_game(rotation=1)
    game.apply(next(move for move in game.legal_moves() if 'Scout' in move))
    state = json.loads(game.to_json())
    state['players'][0]['money'] = 65535
    return brass.Game.from_json(json.dumps(state))


def swap_private_hand_with_deck(game, seat=0):
    """Conservation-preserving hidden-card swap for public-feature invariance."""
    import brass
    import json
    state = json.loads(game.to_json())
    state['players'][seat]['hand'][0], state['deck'][0] = state['deck'][0], state['players'][seat]['hand'][0]
    return brass.Game.from_json(json.dumps(state))
