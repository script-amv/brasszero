# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import Iterable, List

from .arena import GameResult


def write_results(path: str, results: Iterable[GameResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "seed": r.seed,
                        "seats": list(r.seats),
                        "ranking": list(r.ranking),
                        "vp": list(r.vp),
                        "moves": r.moves,
                    }
                )
                + "\n"
            )


def read_results(paths: Iterable[str]) -> List[GameResult]:
    out = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                out.append(
                    GameResult(
                        seed=o["seed"],
                        seats=tuple(o["seats"]),
                        ranking=tuple(o["ranking"]),
                        vp=tuple(o["vp"]),
                        moves=o["moves"],
                    )
                )
    return out
