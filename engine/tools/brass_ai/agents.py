from typing import Callable, Dict, List

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

import brass


class Agent(Protocol):
    """A player that sees only its `Observation`. The seed its factory receives must never be
    derivable into a game seed (callers derive agent seeds by one-way hashing)."""

    name: str

    def act(self, obs: "brass.Observation") -> int: ...


_REGISTRY: Dict[str, Callable[[int], Agent]] = {}


def register(name: str, factory: Callable[[int], Agent]) -> None:
    _REGISTRY[name] = factory


def is_known(spec: str) -> bool:
    return spec in _REGISTRY


def known_agents() -> List[str]:
    return sorted(_REGISTRY)


def make_agent(spec: str, seed: int) -> Agent:
    try:
        factory = _REGISTRY[spec]
    except KeyError:
        raise KeyError(
            "unknown agent %r; known agents: %s" % (spec, ", ".join(known_agents()))
        ) from None
    return factory(seed)


register("random", brass.RandomBot)
register("greedy", brass.GreedyBot)
