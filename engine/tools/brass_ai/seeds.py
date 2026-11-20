"""Seed derivation. Every derived seed is a one-way hash (BLAKE2b) of a domain tag and its
inputs, so no derived seed can be inverted into the master seed, another deal's seed, or
(for agent seeds) the seed of the game the agent is playing."""
import hashlib
import secrets
from typing import List

MASK = 0xFFFF_FFFF_FFFF_FFFF


def hash64(*parts: object) -> int:
    """A 64-bit one-way hash of `parts`, joined with '|'."""
    data = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def game_seeds(master: int, n: int) -> List[int]:
    """`n` deal seeds, reproducible from `master`; no deal seed reveals `master` or another deal's seed."""
    master &= MASK
    return [hash64("deal", master, i) for i in range(n)]


def match_agent_seed(master: int, deal_index: int, rotation: int, seat: int) -> int:
    """The seed for the agent in `seat` of rotation `rotation` of deal `deal_index` of a match."""
    return hash64("agent", master & MASK, deal_index, rotation, seat)


def play_agent_seed(game_seed: int, seat: int) -> int:
    """The seed for the bot in `seat` of a single `play` game; one-way, so it hides `game_seed`."""
    return hash64("play-agent", game_seed & MASK, seat)


def fresh_seed() -> int:
    return secrets.randbits(64)
