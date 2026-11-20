# SPDX-License-Identifier: Apache-2.0
"""Command line: `python -m brass_ai match|rate|play`."""
import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

from .agents import is_known, known_agents, make_agent
from .arena import AgentError, GameResult, play_game, run_match
from .human import HumanAgent, QuitGame
from .rating import fit_ratings
from .results import read_results, write_results
from .seeds import fresh_seed, play_agent_seed
from .stats import summarize

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ci(ci, fmt):
    return ("[" + fmt + ", " + fmt + "]") % ci


def format_report(results: Sequence[GameResult], anchor: str = "random") -> str:
    ratings = fit_ratings(results, anchor)
    used = anchor if anchor in ratings or not ratings else min(ratings)  # as fit_ratings
    header = (
        "agent", "seat-games", "win%", "[95% CI]", "mean-rank", "[CI]", "VP-margin", "[CI]",
        "rating (anchor: %s)" % used,
    )
    rows = [header]
    for s in summarize(results):
        rows.append(
            (
                s.name,
                str(s.seat_games),
                "%.1f" % (100 * s.win_rate),
                _ci((100 * s.win_ci[0], 100 * s.win_ci[1]), "%.1f"),
                "%.2f" % s.mean_rank,
                _ci(s.rank_ci, "%.2f"),
                "%+.1f" % s.vp_margin,
                _ci(s.margin_ci, "%.1f"),
                "%+.0f" % ratings[s.name],
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = []
    for r in rows:
        lines.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))))
    return "\n".join(lines)


def play_human(seat: int, opponents: Sequence[str], seed: int) -> None:
    """One game with a human in `seat`; the seed is only revealed at the end."""
    others = iter(opponents)
    agents = [
        HumanAgent() if s == seat else make_agent(next(others), play_agent_seed(seed, s))
        for s in range(4)
    ]

    def on_move(p, obs, i):
        if p != seat:
            print("P%d (%s): %s" % (p, agents[p].name, obs.describe_move(i)))

    try:
        result = play_game(agents, seed, on_move=on_move)
    except (QuitGame, KeyboardInterrupt):
        print("Game abandoned.")
        return
    except AgentError as e:
        if isinstance(e.__cause__, QuitGame):
            print("Game abandoned.")
            return
        raise
    print("Game over.")
    order = sorted(range(4), key=lambda s: (result.ranking[s], s))
    for s in order:
        print("%d. P%d (%s): %d VP" % (result.ranking[s] + 1, s, result.seats[s], result.vp[s]))
    print("seed: %d" % seed)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="brass_ai")
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match", help="play a seat-rotated match and report")
    m.add_argument("--agents", nargs=4, required=True, metavar="SPEC")
    m.add_argument("--deals", type=int, required=True)
    m.add_argument("--seed", type=int, default=None)
    m.add_argument("--out", default=None)
    m.add_argument("--append", action="store_true", help="add to an existing --out file")
    r = sub.add_parser("rate", help="summarize and rate saved results")
    r.add_argument("files", nargs="+", metavar="FILE")
    r.add_argument("--anchor", default="random")
    pl = sub.add_parser("play", help="play a game against bots in the terminal")
    pl.add_argument("--seat", type=int, required=True, choices=range(4))
    pl.add_argument("--opponents", nargs=3, required=True, metavar="SPEC")
    pl.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    specs = args.agents if args.cmd == "match" else args.opponents if args.cmd == "play" else []
    unknown = [s for s in specs if not is_known(s)]
    if unknown:
        parser.error(
            "unknown agent %s; known agents: %s"
            % (", ".join(repr(s) for s in unknown), ", ".join(known_agents()))
        )

    if args.cmd == "match":
        seed = fresh_seed() if args.seed is None else args.seed
        out = args.out or str(RESULTS_DIR / ("match-%d.jsonl" % seed))
        if os.path.exists(out) and not args.append:
            parser.error("%s already exists; pass --append to add to it, or choose another --out" % out)
        print("seed: %d" % seed)
        results = run_match(args.agents, args.deals, seed)
        write_results(out, results)
        print("results: %s" % out)
        print(format_report(results))
    elif args.cmd == "play":
        play_human(args.seat, args.opponents, fresh_seed() if args.seed is None else args.seed)
    else:
        print(format_report(read_results(args.files), args.anchor))


if __name__ == "__main__":
    main()
