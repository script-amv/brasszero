# SPDX-License-Identifier: Apache-2.0
"""Public-event chunk alignment, target joins and detached rollout boundaries."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from brass_ai.arena import GameResult
from brass_ai.events import PublicEvent
from brass_rl.league import Lineup, LineupEntry
from brass_rl.rollout import Decision, Episode
from brass_rl.returns import DecisionTarget
from helpers import native_game


def boundary_episode(game_id=7):
    obs = native_game().observation(0)
    events = tuple(PublicEvent(k, 0, '{}', str(k)) for k in range(1, 131))
    states = tuple(torch.full((1, 2), float(k), requires_grad=True) for k in range(131))
    decisions = tuple(Decision(game_id, k, 0, obs, 0, -.5, (.2, .3, .1, .4))
                      for k in (0, 63, 64, 65, 129))
    lineup = Lineup((LineupEntry('current', trainable=True),) * 4)
    episode = Episode(game_id, '0', events, {0: states}, decisions, lineup,
                      GameResult(1, ('a', 'b', 'c', 'd'), (0, 1, 2, 3), (4, 3, 2, 1), 130))
    targets = {d.key: DecisionTarget(np.full(4, d.ply, dtype=np.float32), d.ply, True)
               for d in reversed(decisions)}
    return episode, targets


def test_event_boundary_alignment():
    from brass_rl.sequences import sequence_batches
    episode, targets = boundary_episode()
    batches = sorted(sequence_batches([episode], targets, 64, 16, random.Random(4)), key=lambda b: b.start_ply)
    assert [b.decision_positions for b in batches] == [(0, 63), (0, 1), (1,)]
    assert [[d.ply for d in b.decisions] for b in batches] == [[0, 63], [64, 65], [129]]
    for batch in batches:
        for decision, position, target in zip(batch.decisions, batch.decision_positions, batch.targets):
            assert decision.ply == batch.start_ply + position
            assert target.value_targets[0] == decision.ply  # key join, not map insertion order
        assert batch.main_events[0].ply == batch.start_ply + 1


def test_burnin_no_loss_and_no_cross_game_state():
    from brass_rl.sequences import sequence_batches
    first, targets = boundary_episode()
    second, other_targets = boundary_episode(8)
    second = replace(second, boundary_states={0: tuple(t.detach()+1000 for t in second.boundary_states[0])})
    targets.update(other_targets)
    batches = list(sequence_batches([first, second], targets, 64, 16, random.Random(1)))
    assert len(batches) == 6
    for batch in batches:
        s, a = batch.burn_start_ply, batch.start_ply
        assert s == max(0, a-16)
        assert [e.ply for e in batch.burn_in_events] == list(range(s+1, a+1))
        assert [e.ply for e in batch.main_events] == list(range(a+1, batch.end_ply+1))
        assert all(a <= d.ply < batch.end_ply for d in batch.decisions)
        assert batch.previous_public_json == str(s)
        assert not batch.recorded_start_state.requires_grad
        assert batch.recorded_start_state[0, 0] == s + (1000 if batch.game_id == 8 else 0)
    batches[0].recorded_start_state.add_(23)
    assert first.boundary_states[0][0][0, 0] == 0


def test_bad_chronology_and_missing_targets_rejected():
    from brass_rl.sequences import sequence_batches
    ep, targets = boundary_episode()
    for bad in (replace(ep, events=(replace(ep.events[0], ply=2),)+ep.events[1:]),
                replace(ep, decisions=(replace(ep.decisions[0], ply=130),))):
        with pytest.raises(ValueError):
            list(sequence_batches([bad], targets, 64, 16, random.Random(1)))
    with pytest.raises(ValueError, match='target'):
        list(sequence_batches([ep], {}, 64, 16, random.Random(1)))
