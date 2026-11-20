"""Run one recorded four-seat native game using non-secret JSON profile files."""

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from brass_ai.agents import make_agent
from brass_ai.arena import AgentError, play_game
from brass_ai.replay_archive import Archive, recording_context
from brass_ai.seeds import fresh_seed, play_agent_seed

from .protocol import LLMError, Profile
from .registry import ProviderRegistry


def _read_profile(path):
    try:
        return Profile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, RecursionError):
        pass
    raise LLMError("profile")


def main(argv=None, *, registry=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seats",
        nargs=4,
        required=True,
        metavar="SEAT",
        help="For each seat, provide a profile JSON path, random, or greedy.",
    )
    parser.add_argument(
        "--key-env",
        default="OPENROUTER_API_KEY",
        metavar="VARIABLE",
        help="Name of the environment variable holding the selected profiles’ credential.",
    )
    parser.add_argument(
        "--seed", type=int, help="Unsigned 64-bit deal seed. Defaults to a fresh deal."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Replay directory. Defaults to the project archive.",
    )
    args = parser.parse_args(argv)
    seed = fresh_seed() if args.seed is None else args.seed
    if not 0 <= seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    registry = registry or ProviderRegistry()
    agents = []
    game_id = uuid.uuid4().hex
    try:
        for seat, spec in enumerate(args.seats):
            if spec in ("random", "greedy"):
                agents.append(make_agent(spec, play_agent_seed(seed, seat)))
            else:
                profile = _read_profile(spec)
                agents.append(
                    registry.make_agent(profile, os.environ.get(args.key_env))
                )
        with recording_context(
            category="terminal", interface="llm-arena", session=game_id
        ):
            result = play_game(
                agents, seed, archive=Archive(args.archive), game_id=game_id
            )
        print(json.dumps(dict(asdict(result), game_id=game_id)))
        return 0
    except (AgentError, LLMError) as error:
        # The arena wraps hook and act failures. Only publish our fixed error vocabulary.
        safe = error if isinstance(error, LLMError) else error.__cause__
        print(
            str(safe) if isinstance(safe, LLMError) else "The native game stopped.",
            file=sys.stderr,
        )
        return 1
    finally:
        for agent in agents:
            if hasattr(agent, "close"):
                agent.close()
