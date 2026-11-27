#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export one recorded game as public replay data for the website.

Replays the recorded moves through the native engine and writes, for every
decision, the public board state after it: built tiles, built links, and each
player's victory points, income level and money. Requires the `brass` native
binding (`make setup`).

    python3 site/scripts/export_replay.py replays/games/<id>.json.gz --slug record-124

The output holds no deal seed, hands, deck order or model weights. Cards that a
move discards are public in the game and are omitted here as well.
"""

import argparse
import datetime
import gzip
import json
from pathlib import Path

import brass

INDUSTRIES = ["cotton", "manufacturer", "pottery", "coal", "iron", "brewery"]
OUT = Path(__file__).resolve().parents[1] / "public/data/replays"


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def place(location_id):
    words = location_id.replace("-", " ").title().replace(" On Trent", "-on-Trent")
    return "Farm brewery" if location_id.startswith("farm-") else words


def describe(move, board):
    if isinstance(move, str):
        return {"EndAction": "End of turn"}.get(move, move)
    kind, body = next(iter(move.items()))
    town = lambda slot: board["names"][board["slots"][slot]["loc"]]  # noqa: E731
    link = lambda i: "{} – {}".format(*(board["names"][board["links"][i][k]] for k in ("a", "b")))  # noqa: E731
    if kind == "Build":
        return f"Build {body['industry']} in {town(body['slot'])}"
    if kind in ("Develop", "DevelopSecond", "BonusDevelop"):
        return f"Develop {body['industry']}"
    if kind in ("Sell", "SellAnother"):
        return f"Sell from {town(body['slot'])}"
    if kind == "Canal":
        return f"Canal {link(body['link'])}"
    if kind in ("Rail", "SecondRail"):
        return f"Rail {link(body['link'])}"
    if kind == "ShortfallRemove":
        return f"Remove tile in {town(body['slot'])} to cover a shortfall"
    return {"Loan": "Take a loan", "Scout": "Scout for wild cards", "Pass": "Pass"}.get(kind, kind)


def target(move):
    """The slot or link a move changes, for highlighting on the board."""
    if isinstance(move, str):
        return None
    body = next(iter(move.values()))
    if "slot" in body:
        return {"slot": body["slot"]}
    if "link" in body:
        return {"link": body["link"]}
    return None


def snapshot(game, income_levels):
    pub = json.loads(game.observation(0).public_json())
    slots = [
        [s["owner"], INDUSTRIES.index(s["industry"]), s["level"], int(s["flipped"])] if s else None
        for s in pub["slots"]
    ]
    players = [[p["vp"], income_levels[p["income_space"]], p["money"]] for p in pub["players"]]
    return {"era": pub["era"], "round": pub["round"], "slots": slots, "links": pub["links"], "players": players}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", help="recorded game file (.json or .json.gz)")
    parser.add_argument("--slug", required=True, help="output file name without extension")
    parser.add_argument("--featured", type=int, help="seat to feature (default: the winner)")
    args = parser.parse_args()

    record = load(args.record)
    data = json.loads(brass.component_data())
    board = {
        "names": [place(loc["id"]) for loc in data["locations"]],
        "slots": data["slots"],
        "links": data["links"],
    }
    game = brass.Game(record["initial"]["seed"])
    states = [snapshot(game, data["income_levels"])]
    plies = []
    for entry in record["moves"]:
        game.apply(json.dumps(entry["move"]))
        state = snapshot(game, data["income_levels"])
        plies.append({"p": entry["player"], "text": describe(entry["move"], board), "end": entry["move"] == "EndAction",
                      "target": target(entry["move"])})
        states.append(state)
    if not game.is_over() or list(game.vp()) != record["result"]["vp"]:
        raise SystemExit("replay does not reproduce the recorded result")

    updates = record.get("provenance", {}).get("seat_updates") or [None] * 4
    out = {
        "meta": {
            "game": record["game_id"],
            "category": record["category"],
            "exported": datetime.date.today().isoformat(),
            "featured": args.featured if args.featured is not None else record["result"]["winner"],
            "seats": [f"Update {u}" if u is not None else f"Seat {i + 1}" for i, u in enumerate(updates)],
            "final_vp": record["result"]["vp"],
            "ranking": record["result"]["ranking"],
        },
        "board": {
            "locations": [{"name": place(loc["id"]), "id": loc["id"], "kind": loc["kind"], "slots": loc["slots"]}
                          for loc in data["locations"]],
            "links": [[ln["a"], ln["b"], int(ln["canal"]), int(ln["rail"])] for ln in data["links"]],
            "industries": INDUSTRIES,
        },
        "plies": plies,
        "states": states,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.slug}.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"Wrote {len(plies)} moves to {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
