"""Pure controller seed streams. Derived agent seeds never contain a deal seed."""
from dataclasses import dataclass

from brass_ai.seeds import hash64


@dataclass(frozen=True)
class SeedStreams:
    master: int

    def __post_init__(self):
        if type(self.master) is not int or not 0 <= self.master < 2**64:
            raise ValueError("master must be an unsigned 64-bit integer")

    def seed(self, domain: str, *ids: int) -> int:
        """Hash namespace/master/IDs independently; no advancing RNG state.

        Controllers use train.deal/dev.deal/final.deal, policy, opponents and
        bootstrap. Additional nonempty domains without delimiters are supported.
        High-entropy masters are required for resistance to seed guessing.
        """
        if not isinstance(domain, str) or not domain or "|" in domain:
            raise ValueError("seed domain must be nonempty and contain no '|'")
        if any(type(i) is not int or not 0 <= i < 2**64 for i in ids):
            raise ValueError("seed IDs must be unsigned 64-bit integers")
        return hash64("brass-rl-v1", domain, self.master, len(ids), *ids)
