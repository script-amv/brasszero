# SPDX-License-Identifier: Apache-2.0
"""Native lifecycle tests, including GUI-style replay of earlier own moves."""
from dataclasses import replace
import json

import brass
import pytest
import torch

from brass_ai.arena import RefereeGame, play_game
from brass_rl.agent import TorchAgent
from brass_rl.config import ModelConfig
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(20)
    c = ComponentData.load()
    return c, BrassPolicy(ModelConfig(), c.schema, c)


def test_agent_requires_complete_ordered_history(setup):
    c, model = setup
    agent = TorchAgent(model, c, 81)
    referee = RefereeGame(52, [brass.RandomBot(i) for i in range(4)])
    seat, obs = referee.pending()
    with pytest.raises(RuntimeError, match="reset"):
        agent.act(obs)
    agent.reset((seat + 1) % 4, obs.public_json())
    with pytest.raises(ValueError, match="viewer|seat"):
        agent.act(obs)
    agent.reset(seat, obs.public_json())
    index = agent.act(obs)
    with pytest.raises(RuntimeError, match="event"):
        agent.act(obs)
    event = referee.apply_index(index)
    with pytest.raises(ValueError, match="ply|order"):
        agent.observe_public(replace(event, ply=2))
    with pytest.raises(ValueError, match="chosen|move"):
        agent.observe_public(replace(event, move_json='"EndAction"'))
    agent.observe_public(event)
    assert not torch.equal(agent.memory_snapshot(), torch.zeros((1, 64)))
    with pytest.raises(ValueError, match="ply|order"):
        agent.observe_public(event)
    with pytest.raises(ValueError, match="snapshot|public"):
        agent.prepare(obs)
    agent.reset(seat, obs.public_json())
    assert torch.equal(agent.memory_snapshot(), torch.zeros((1, 64)))


def test_historical_prefix_reconstructs_including_own_moves_and_era(setup):
    c, model = setup
    reference = TorchAgent(model, c, 12, mode="argmax")
    ref = RefereeGame(44, [reference] + [brass.RandomBot(i) for i in range(1, 4)])
    initial = brass.Game(44).observation(0).public_json()
    events = []
    while not ref.is_over():
        seat, obs = ref.pending()
        events.append(ref.apply_index(reference.act(obs) if seat == 0 else 0))
        if json.loads(events[-1].public_json)["era"] == "rail":
            break
    replay = TorchAgent(model, c, 12, mode="argmax")
    replay.reset(0, initial)
    for event in events:
        replay.observe_public(event)
    torch.testing.assert_close(replay.memory_snapshot(), reference.memory_snapshot(), atol=0, rtol=0)
    snapshot = replay.memory_snapshot()
    snapshot.zero_()
    assert replay.memory_snapshot().abs().sum() > 0
    while not ref.is_over():
        seat, obs = ref.pending()
        if seat == 0:
            assert replay.act(obs) == reference.act(obs)
            break
        event = ref.apply_index(0)
        replay.observe_public(event)


def test_initial_sampling_deterministic(setup):
    c, model = setup
    game = brass.Game(5)
    seat = game.current_player()
    obs = game.observation(seat)
    agents = [TorchAgent(model, c, 900) for _ in range(3)]
    global_rng = torch.random.get_rng_state().clone()
    for agent in agents:
        agent.reset(seat, obs.public_json())
    def decide(agent):
        prepared = agent.prepare(obs)
        out = model([prepared.encoded], prepared.memory, 127)
        return agent.choose(out, 0)
    first = decide(agents[0])
    assert first == decide(agents[1])
    assert torch.equal(global_rng, torch.random.get_rng_state())
    torch.rand(77)
    assert decide(agents[2]).index == first.index
    # Only the deliberate draw above advances the process-global generator.
    torch.random.set_rng_state(global_rng)
    before = torch.random.get_rng_state().clone()
    agent = TorchAgent(model, c, 900, mode="argmax")
    agent.reset(seat, obs.public_json())
    p = agent.prepare(obs)
    out = model([p.encoded], p.memory, 127)
    choice = agent.choose(out, 0)
    assert choice.index == out.logits.argmax().item()
    assert isinstance(choice.log_prob, float) and len(choice.values) == 4
    assert torch.equal(before, torch.random.get_rng_state())


def test_batched_prepare_choose_matches_act(setup):
    c, model = setup
    game = brass.Game(77)
    seat = game.current_player()
    obs = game.observation(seat)
    agents = [TorchAgent(model, c, s) for s in (3, 6)]
    direct = [TorchAgent(model, c, s) for s in (3, 6)]
    for a in agents + direct:
        a.reset(seat, obs.public_json())
    prepared = [a.prepare(obs) for a in agents]
    out = model([p.encoded for p in prepared], torch.cat([p.memory for p in prepared]), 41)
    assert [a.choose(out, i).index for i, a in enumerate(agents)] == [a.act(obs) for a in direct]


def test_untrained_full_seeded_game_consumes_terminal_history(setup):
    c, model = setup
    agent = TorchAgent(model, c, 2026)
    result = play_game([agent] + [brass.RandomBot(i) for i in range(1, 4)], seed=20261117)
    assert result.moves > 100
    assert agent.ply == result.moves
    assert torch.isfinite(agent.memory_snapshot()).all()
    assert json.loads(agent.latest_public_json)["game_over"]


def test_public_history_annotation_cannot_attest_an_altered_observation(setup):
    c, model = setup
    game = brass.Game(5)
    seat = game.current_player()
    agent = TorchAgent(model, c, 562)
    agent.reset(seat, game.observation(seat).public_json())
    changed = json.loads(game.to_json())
    changed["players"][seat]["money"] += 1
    stale = brass.Game.from_json(json.dumps(changed)).observation(seat)
    agent.latest_public_json = stale.public_json()
    with pytest.raises(ValueError, match="public snapshot"):
        agent.prepare(stale)
