# SPDX-License-Identifier: Apache-2.0
"""Contiguous public-event minibatches; only whole sequences may be shuffled."""
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from brass_ai.events import PublicEvent
from .returns import DecisionTarget
from .rollout import Decision, Episode


@dataclass(frozen=True)
class SequenceBatch:
    game_id: int
    viewer: int
    burn_start_ply: int
    start_ply: int
    end_ply: int
    recorded_start_state: Tensor
    previous_public_json: str
    burn_in_events: Tuple[PublicEvent, ...]
    main_events: Tuple[PublicEvent, ...]
    decisions: Tuple[Decision, ...]
    decision_positions: Tuple[int, ...]
    targets: Tuple[DecisionTarget, ...]


def sequence_batches(episodes: Sequence[Episode], targets: Mapping,
                     sequence_events: int, burn_in_events: int, rng) -> Iterator[SequenceBatch]:
    """Join targets by key and shuffle complete seat/chunk records, never steps.

    For loss interval [a,b), replay detached recorded h[s], s=max(0,a-burn),
    then events s+1..a without gradients. Decision k sees h[k] before event k+1.
    Empty-loss chunks are omitted: later chunks still start from their own exact
    recorded boundary, including all intervening opponent events in burn/main.
    """
    if type(sequence_events) is not int or sequence_events < 1:
        raise ValueError('sequence_events must be a positive integer')
    if type(burn_in_events) is not int or not 0 <= burn_in_events <= 16:
        raise ValueError('burn_in_events must be an integer in [0,16]')
    batches, game_ids = [], set()
    for episode in episodes:
        if episode.game_id in game_ids:
            raise ValueError('episode game IDs must be unique')
        game_ids.add(episode.game_id)
        n = len(episode.events)
        if any(type(e.ply) is not int or e.ply != k+1 for k, e in enumerate(episode.events)):
            raise ValueError('episode public events must have consecutive plies starting at 1')
        by_seat = {}
        for decision in episode.decisions:
            if (decision.game_id != episode.game_id or type(decision.seat) is not int
                    or decision.seat not in range(4) or type(decision.ply) is not int
                    or not 0 <= decision.ply < n):
                raise ValueError('decision identity/ply must lie before an episode event')
            entry = episode.lineup.entries[decision.seat]
            if entry.kind != 'current' or not entry.trainable:
                raise ValueError('sequence decisions must belong to current/trainable seats')
            trajectory = by_seat.setdefault(decision.seat, [])
            if trajectory and trajectory[-1].ply >= decision.ply:
                raise ValueError('own-seat decision plies must strictly increase')
            if decision.key not in targets:
                raise ValueError('missing decision target for %s' % (decision.key,))
            trajectory.append(decision)
        for seat, decisions in by_seat.items():
            states = episode.boundary_states.get(seat, ())
            if len(states) != n+1:
                raise ValueError('current seat needs recorded h[0] through h[T]')
            for a in range(0, n, sequence_events):
                b = min(a+sequence_events, n)
                selected = tuple(d for d in decisions if a <= d.ply < b)
                if not selected:
                    continue
                s = max(0, a-burn_in_events)
                state = states[s]
                if (not isinstance(state, Tensor) or state.ndim != 2 or state.shape[0] != 1
                        or state.dtype != torch.float32 or not torch.isfinite(state).all().item()):
                    raise ValueError('recorded boundary must be finite float32 [1,H]')
                batches.append(SequenceBatch(episode.game_id, seat, s, a, b, state.detach().clone(),
                    episode.initial_public_json if s == 0 else episode.events[s-1].public_json,
                    tuple(episode.events[s:a]), tuple(episode.events[a:b]), selected,
                    tuple(d.ply-a for d in selected), tuple(targets[d.key] for d in selected)))
    rng.shuffle(batches)
    yield from batches
