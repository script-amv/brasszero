#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the public training-record data for the website.

Reads the local GUI server's read-only dashboard API (which reduces the replay
archive and finishes the strategy analysis) and writes a compact JSON file to
site/public/data/training-record.json. Standard library only.

    python3 site/scripts/export_training_record.py --url http://127.0.0.1:8773

The output holds no game seeds, actions or model weights: only per-seat scores
with their model update, and window aggregates of the dashboard's metrics.
"""

import argparse
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ACTIONS = ["Build", "Develop", "Sell", "Canal", "Rail", "Loan", "Scout", "Pass"]
INDUSTRIES = ["cotton", "manufacturer", "pottery", "coal", "iron", "brewery"]
ERAS = ["canal", "rail"]
SOURCES = INDUSTRIES + ERAS + ["Merchant", "Losses"]
CATEGORY = {"training": "training", "evaluation": "evaluation"}
WINDOW = 40  # updates per window for behaviour metrics
SCORE_WINDOW = 20  # updates per window for the mean score line


def fetch(base):
    query = urllib.parse.urlencode({"category": "all", "opponents": "all", "cohort": "", "model": ""})
    with urllib.request.urlopen(f"{base}/api/dashboard?{query}", timeout=600) as response:
        return json.load(response)


def weighted(points, key, sub=None):
    num = den = 0.0
    for p in points:
        strategy = p.get("strategy") or {}
        cell = strategy.get(key) if sub is None else (strategy.get(key) or {}).get(sub)
        if cell and cell.get("value") is not None:
            num += cell["value"] * cell["samples"]
            den += cell["samples"]
    return round(num / den, 3) if den else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, help="verified local dashboard export")
    parser.add_argument("--url", default="http://127.0.0.1:8773", help="local GUI server")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "public/data/training-record.json")
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text()) if args.input else fetch(args.url.rstrip("/"))
    if not data.get("data_date"):
        raise ValueError("dashboard export requires newest eligible record date; use make site-data")
    pending = {k: v for k, v in (data["coverage"].get("analysis") or {}).items() if k != "ready"}
    if any(pending.values()):
        raise SystemExit(f"Strategy analysis not finished yet: {pending}. Retry in a minute.")

    games = [g for g in data["games"] if g["update"] is not None and g.get("model_id")]
    if any(g.get("category") in {"qa", "validation", "test"} for g in data["games"]):
        raise ValueError("QA/validation row in dashboard input")
    if any(type(g["update"]) is not int or g["update"] < 0 for g in games):
        raise ValueError("invalid known model update")
    latest = max(g["update"] for g in games)

    # Every AI seat score with a known model update.
    scores = sorted([g["update"], g["vp"], CATEGORY.get(g["category"], "other")] for g in games)

    # Mean score per SCORE_WINDOW updates (complete windows only).
    by_window = defaultdict(list)
    for update, vp, _ in scores:
        by_window[update // SCORE_WINDOW * SCORE_WINDOW].append(vp)
    score_mean = [
        {"lo": lo, "hi": lo + SCORE_WINDOW - 1, "n": len(v), "mean": round(sum(v) / len(v), 2)}
        for lo, v in sorted(by_window.items())
        if lo + SCORE_WINDOW - 1 <= latest
    ]

    # Behaviour metrics per WINDOW updates, from the dashboard's per-policy points.
    points = defaultdict(list)
    for p in data["points"]:
        if p["update"] is not None:
            points[p["update"] // WINDOW * WINDOW].append(p)
    windows = []
    for lo, ps in sorted(points.items()):
        if lo + WINDOW - 1 > latest:
            continue  # incomplete window
        actions = Counter()
        for p in ps:
            actions.update(p["actions"])
        total = sum(actions[a] for a in ACTIONS)
        with_sources = [p["sources"] for p in ps if p.get("sources")]
        samples = {}
        for metric, keys in [
            ("loans", ERAS),
            ("income", ERAS),
            ("network", ERAS),
            ("unflipped", ERAS),
            ("built", INDUSTRIES),
        ]:
            samples[metric] = {
                k: sum(((p.get("strategy") or {}).get(metric, {}).get(k) or {}).get("samples", 0) for p in ps)
                for k in keys
            }
        samples["scored_twice"] = {
            "value": sum(((p.get("strategy") or {}).get("twice_vp") or {}).get("samples", 0) for p in ps)
        }
        samples["actions"] = {a: total for a in ACTIONS}
        samples["sources"] = {k: len(with_sources) for k in SOURCES}
        windows.append(
            {
                "samples": samples,
                "lo": lo,
                "hi": lo + WINDOW - 1,
                "policies": len(ps),
                "seat_games": sum(p["samples"] for p in ps),
                "actions": {a: round(100 * actions[a] / total, 2) for a in ACTIONS} if total else None,
                "loans": {e: weighted(ps, "loans", e) for e in ERAS},
                "income": {e: weighted(ps, "income", e) for e in ERAS},
                "unflipped": {e: weighted(ps, "unflipped", e) for e in ERAS},
                "network": {e: weighted(ps, "network", e) for e in ERAS},
                "built": {i: weighted(ps, "built", i) for i in INDUSTRIES},
                "scored_twice": weighted(ps, "twice_vp"),
                "sources": {
                    s: round(sum(x.get(s, 0) for x in with_sources) / len(with_sources), 2) for s in SOURCES
                }
                if with_sources
                else None,
            }
        )

    record = max(games, key=lambda g: (g["vp"], -g["update"]))
    out = {
        "meta": {
            "exported": data["data_date"],
            "source": "Replay archive via the local GUI dashboard (all categories, all opponents)",
            "games": data["game_count"],
            "ai_seat_scores": len(scores),
            "latest_update": latest,
            "window": WINDOW,
            "score_window": SCORE_WINDOW,
        },
        "audit": {
            "known_update_counts": sorted(Counter(g["update"] for g in games).items()),
            "unknown_update_seats": sum(g["update"] is None for g in data["games"]),
            "qa_rows": 0,
        },
        "scores": scores,
        "score_mean": score_mean,
        "windows": windows,
        "record": {
            "vp": record["vp"],
            "update": record["update"],
            "label": record["label"],
            "category": record["category"],
            "opponents": record["opponent_kind"],
            "game_id": record["game_id"],
            "seat": record["seat"],
            "sources": (record.get("strategy") or {}).get("sources"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(
        f"Wrote {args.out}: {len(scores)} scores, {len(windows)} windows, latest update {latest}, record {record['vp']} VP"
    )


if __name__ == "__main__":
    main()
