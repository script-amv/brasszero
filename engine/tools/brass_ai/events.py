"""Immutable public events shared by every seat, with no deal or private state."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicEvent:
    """One successful native apply; ply starts at one and includes the terminal move."""

    ply: int
    actor: int
    move_json: str
    public_json: str
