# SPDX-License-Identifier: Apache-2.0
"""Stateless lineup sampling and a bounded, immutable opponent snapshot pool.

Only retained training snapshots plus an incoming snapshot participate in retention.
Evicted weights need not survive. Historical ID/digest metadata prevents ID reuse;
development holdouts are retained separately and never enter sampled lineups.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Optional, Tuple

from .config import LeagueConfig
from .seeding import SeedStreams


@dataclass(frozen=True)
class Snapshot:
    id: str
    weights_path: Path
    update: int
    role: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("snapshot ID must be nonempty")
        if type(self.update) is not int or self.update < 0:
            raise ValueError("snapshot update must be a nonnegative integer")
        if self.role not in ("initial", "training", "holdout"):
            raise ValueError("snapshot role must be initial, training or holdout")
        object.__setattr__(self, "weights_path", Path(self.weights_path).resolve())


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class LineupEntry:
    kind: str
    snapshot_id: Optional[str] = None
    trainable: bool = False

    def __post_init__(self):
        if self.kind not in ("current", "frozen", "random", "greedy"):
            raise ValueError("unknown lineup kind")
        if type(self.trainable) is not bool or self.trainable != (self.kind == "current"):
            raise ValueError("only current seats are trainable")
        if (self.snapshot_id is not None) != (self.kind == "frozen"):
            raise ValueError("only frozen seats have a snapshot ID")


@dataclass(frozen=True)
class Lineup:
    entries: Tuple[LineupEntry, ...]

    def __post_init__(self):
        entries = tuple(self.entries)
        if len(entries) != 4 or not all(isinstance(e, LineupEntry) for e in entries):
            raise ValueError("lineup needs four entries")
        if not any(e.trainable for e in entries):
            raise ValueError("lineup needs a current seat")
        object.__setattr__(self, "entries", entries)


class League:
    def __init__(self, streams: SeedStreams, initial: Snapshot,
                 config: LeagueConfig = LeagueConfig(), holdouts=()):
        if initial.role != "initial":
            raise ValueError("league requires an initial snapshot")
        self.streams, self.config = streams, config
        self._initial = initial
        self._pool = ()  # type: Tuple[Snapshot, ...]
        self._holdouts = ()  # type: Tuple[Snapshot, ...]
        self._known = {}  # type: Dict[str, Tuple[Snapshot, str]]
        self._history = {}  # ID -> lightweight immutable metadata, including evictions
        self.add(initial)
        for snapshot in holdouts:
            if snapshot.role != "holdout":
                raise ValueError("holdouts must have holdout role")
            self.add(snapshot)

    @property
    def snapshots(self) -> Tuple[Snapshot, ...]:
        return self._pool

    @property
    def holdouts(self) -> Tuple[Snapshot, ...]:
        return self._holdouts

    @property
    def history(self):
        """Path-free ID/digest tombstones; evicted weight files need not exist."""
        return {sid: dict(record) for sid, record in self._history.items()}

    @classmethod
    def restore(cls, streams, snapshots, config=LeagueConfig(), holdouts=(), history=None):
        """Restore exact retained membership without running retention or drawing RNG.

        The artifact loader supplies resolved immutable store paths. Only active
        training/holdout files are opened; historical records are metadata only.
        """
        snapshots, holdouts = tuple(snapshots), tuple(holdouts)
        initials = [s for s in snapshots if s.role == "initial"]
        if (len(initials) != 1 or len(snapshots) > config.max_snapshots or
                any(s.role == "holdout" for s in snapshots) or
                any(s.role != "holdout" for s in holdouts) or
                len({s.id for s in snapshots + holdouts}) != len(snapshots + holdouts)):
            raise ValueError("invalid retained league membership")
        restored = cls(streams, initials[0], config)
        records = {} if history is None else {sid: dict(record) for sid, record in history.items()}
        for sid, record in records.items():
            if (not isinstance(sid, str) or not sid or set(record) != {"update", "role", "sha256"} or
                    type(record["update"]) is not int or record["update"] < 0 or
                    record["role"] not in ("initial", "training", "holdout") or
                    not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or
                    any(c not in "0123456789abcdef" for c in record["sha256"])):
                raise ValueError("invalid league history metadata")
        for snapshot in snapshots + holdouts:
            fingerprint = _digest(snapshot.weights_path)
            record = {"update": snapshot.update, "role": snapshot.role, "sha256": fingerprint}
            if snapshot.id in records and records[snapshot.id] != record:
                raise ValueError("retained snapshot conflicts with immutable history")
            if snapshot.role == "training" and snapshot.update <= initials[0].update:
                raise ValueError("training snapshot must follow initialization")
            records[snapshot.id] = record
            restored._known[snapshot.id] = (snapshot, fingerprint)
        restored._pool, restored._holdouts, restored._history = snapshots, holdouts, records
        return restored

    def verify(self, snapshot: Snapshot) -> None:
        known = self._known.get(snapshot.id)
        if known is None or known[0] != snapshot:
            raise ValueError("snapshot ID metadata conflicts with league")
        if _digest(snapshot.weights_path) != known[1]:
            raise ValueError("immutable snapshot weights changed: " + snapshot.id)

    def add(self, snapshot: Snapshot) -> None:
        fingerprint = _digest(snapshot.weights_path)
        known = self._known.get(snapshot.id)
        if known is not None:
            if known != (snapshot, fingerprint):
                raise ValueError("snapshot ID already refers to different metadata/immutable weights")
            return
        record = {"update": snapshot.update, "role": snapshot.role, "sha256": fingerprint}
        if snapshot.id in self._history:
            if self._history[snapshot.id] != record:
                raise ValueError("snapshot ID already refers to different immutable history")
            return  # Historical identical IDs never re-enter a retained pool.
        if snapshot.role == "initial" and self._known:
            raise ValueError("league already has an initial snapshot")
        if snapshot.role == "training" and snapshot.update <= self._initial.update:
            raise ValueError("training snapshot must follow initialization")
        self._known[snapshot.id] = (snapshot, fingerprint)
        self._history[snapshot.id] = record
        if snapshot.role == "holdout":
            self._holdouts = tuple(sorted(self._holdouts + (snapshot,), key=lambda s: (s.update,s.id)))
            return
        self._pool = self._retain(self._pool + (snapshot,))

    def _retain(self, candidates):
        cap = self.config.max_snapshots
        ordered = sorted(candidates, key=lambda s: (s.update, s.id))
        chosen = [self._initial]
        training = [s for s in ordered if s.id != self._initial.id]
        if training and len(chosen) < cap:
            chosen.append(training[0])
        remaining = [s for s in training if s not in chosen]
        recent = list(reversed(remaining))[:min(3, cap-len(chosen))]
        chosen.extend(recent)
        older = [s for s in remaining if s not in recent]
        slots = min(cap-len(chosen), len(older))
        if slots:
            # Interior equally spaced update targets between the pinned first
            # training snapshot and the latest. Ties favor older update then ID.
            low, high = training[0].update, training[-1].update
            for i in range(1, slots+1):
                target = low + (high-low)*i/(slots+1)
                nearest = min(older,key=lambda s: (abs(s.update-target),s.update,s.id))
                chosen.append(nearest)
                older.remove(nearest)
        return tuple(sorted(chosen,key=lambda s: (s.update,s.id)))

    def sample(self, game_id: int) -> Lineup:
        rng = random.Random(self.streams.seed("opponents", game_id))
        for snapshot in self._pool:
            self.verify(snapshot)
        draw = rng.random()
        self_play, frozen, _ = self.config.lineup_probabilities
        current = LineupEntry("current", trainable=True)
        if self.config.lineup_schedule == "four_of_five":
            # Game IDs alone fix exposure across resumes and preserve the original
            # per-game RNG draws for the focal seat and historical opponents.
            draw = self_play if game_id % 5 == 0 else 0.
        if draw < self_play:
            return Lineup((current,)*4)
        focal = rng.randrange(4)
        eligible = tuple(LineupEntry("frozen", s.id) for s in self._pool)
        if draw >= self_play + frozen:
            eligible += (LineupEntry("random"), LineupEntry("greedy"))
        return Lineup(tuple(current if s == focal else rng.choice(eligible) for s in range(4)))


def replay_identity(snapshot_id, record):
    if snapshot_id.startswith('policy-sha256-v1:'):
        return snapshot_id
    return '%s-%06d:weights-sha256:%s' % (record['role'], record['update'], record['sha256'])
