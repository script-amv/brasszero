# SPDX-License-Identifier: Apache-2.0
"""Independent PPO arithmetic and real recurrent optimizer/rollback behavior."""
from copy import deepcopy
from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch

from brass_ai.arena import GameResult
from brass_ai.events import PublicEvent
from brass_rl.config import LeagueConfig, ModelConfig, PPOConfig, TrainConfig
from brass_rl.distributions import PolicyStats, policy_stats, policy_probabilities
from brass_rl.features import encode_observation, encode_public
from brass_rl.league import League, Lineup, LineupEntry, Snapshot
from brass_rl.model import BrassPolicy
from brass_rl.rollout import Decision, Episode, collect_batch
from brass_rl.returns import DecisionTarget, episode_targets
from brass_rl.schema import ComponentData
from brass_rl.seeding import SeedStreams
from helpers import native_game, shortfall_game


def stats(logp=math.log(.6), values=None, entropy=0.):
    return PolicyStats(torch.tensor([logp], requires_grad=True), torch.tensor([entropy]),
                       torch.zeros((1, 4), requires_grad=True) if values is None else values)


def test_clipped_surrogate_hand_values():
    from brass_rl.ppo import ppo_loss
    for advantage, want in ((1., -1.2), (-1., 1.5)):
        terms = ppo_loss(stats(), [math.log(.4)], [advantage], [[0.]*4], [True], PPOConfig())
        assert terms.policy.item() == pytest.approx(want)
        assert terms.value.item() == 0.
        assert terms.clip_fraction.item() == 1.
        assert terms.approx_kl.item() == pytest.approx(.5-math.log(1.5))
        terms.total.backward()


def test_uniform_two_move_entropy_and_four_component_mse():
    from brass_rl.ppo import ppo_loss
    from brass_rl.distributions import PolicyOutput
    output = PolicyOutput(torch.zeros(2), torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool),
                          torch.tensor([[.1, .2, .3, .4]]))
    terms = ppo_loss(policy_stats(output, torch.tensor([0])), [-math.log(2)], [0.],
                     [[1., 0., 0., 0.]], [True], PPOConfig())
    assert terms.entropy.item() == pytest.approx(math.log(2))
    assert terms.value.item() == pytest.approx(.275)
    assert terms.total.item() == pytest.approx(.5*.275-.01*math.log(2))


def test_singleton_value_only_and_nonfinite_inputs():
    from brass_rl.ppo import ppo_loss
    terms = ppo_loss(stats(0., entropy=0.), [0.], [99.], [[1., 0., 0., 0.]], [False], PPOConfig())
    assert terms.policy.item() == terms.entropy.item() == terms.approx_kl.item() == 0.
    assert terms.value.item() == .25
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError, match='finite'):
            ppo_loss(stats(bad), [0.], [1.], [[0.]*4], [True], PPOConfig())
        with pytest.raises(ValueError, match='finite'):
            ppo_loss(stats(), [0.], [bad], [[0.]*4], [True], PPOConfig())


def test_rollout_wide_normalization_excludes_singletons():
    from brass_rl.ppo import normalize_targets
    targets = {(0, 0, 0): DecisionTarget(np.zeros(4), 1., True),
               (1, 0, 0): DecisionTarget(np.zeros(4), 3., True),
               (1, 1, 0): DecisionTarget(np.zeros(4), 1000., False)}
    normalized = normalize_targets(targets)
    assert [normalized[k].self_advantage for k in targets] == [-1., 1., 0.]
    assert targets[(0, 0, 0)].self_advantage == 1.
    assert normalize_targets({}) == {}
    one = normalize_targets({(0, 0, 0): targets[(0, 0, 0)]})
    assert one[(0, 0, 0)].self_advantage == 0.


def test_episode_targets_nondefault_lambda_zero():
    from test_returns import decision, episode
    first = decision(0, values=(.2, .3, .1, .4))
    second = decision(4, values=(.4, .2, .1, .3))
    target = episode_targets(episode([first, second]), gamma=1., lam=0.)[first.key]
    np.testing.assert_allclose(target.value_targets, [.4, .2, .1, .3], atol=1e-6)
    assert target.self_advantage == pytest.approx(.2, abs=1e-6)


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(120)
    components = ComponentData.load()
    model = BrassPolicy(ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4,
                                     message_passing_rounds=1), components.schema, components)
    return model, components


def toy_episodes(model, components):
    """Native two-choice shortfall observations and real public-history embeddings.

    Terminal labels are synthetic wins/losses for this optimizer fixture only;
    this does not purport to be complete Brass gameplay or strength evidence.
    """
    episodes = []
    for game_id, choice, ranking in ((0, 0, (0, 1, 2, 3)), (1, 1, (1, 0, 2, 3))):
        game = shortfall_game()
        initial = game.observation(0).public_json()
        move = game.observation(0).legal_moves()[0]
        game.apply_index(0)
        obs = game.observation(0)
        event = PublicEvent(1, 0, move, obs.public_json())
        h0 = model.initial_memory(1, model.device)
        with torch.no_grad():
            h1 = model.update_memory([encode_public(event, 0, initial, components)], h0)
            output = model([encode_observation(obs, components)], h1)
            old = policy_stats(output, torch.tensor([choice]))
        assert obs.num_legal() == 2
        decision = Decision(game_id, 1, 0, obs, choice, old.log_prob.item(), tuple(old.values[0].tolist()))
        move = obs.legal_moves()[choice]
        game.apply_index(choice)
        last = PublicEvent(2, 0, move, game.observation(0).public_json())
        with torch.no_grad():
            h2 = model.update_memory([encode_public(last, 0, event.public_json, components)], h1)
        lineup = Lineup((LineupEntry('current', trainable=True), LineupEntry('random'),
                         LineupEntry('greedy'), LineupEntry('random')))
        episodes.append(Episode(game_id, initial, (event, last), {0: (h0, h1, h2)}, (decision,), lineup,
                                GameResult(1, ('a','b','c','d'), ranking, (4,3,2,1), 2)))
    return episodes


def test_tiny_recurrent_two_choice_learning_and_legal_distribution(tiny, tmp_path):
    from brass_rl.ppo import update_policy
    model, components = tiny
    pool = tmp_path/'frozen.pt'
    torch.save(model.state_dict(), pool)
    original = pool.read_bytes()
    config = PPOConfig(learning_rate=.02, epochs=2, target_kl=.03)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    def probability():
        ep = toy_episodes(model, components)[0]
        with torch.no_grad():
            out = model([encode_observation(ep.decisions[0].observation, components)], ep.boundary_states[0][1])
            return policy_probabilities(out)
    before = probability()[0].item()
    gru_before = model.event_gru.weight_ih.detach().clone()
    for update in range(8):
        metrics = update_policy(model, optimizer, toy_episodes(model, components), config, random.Random(update))
        assert metrics.optimizer_steps > 0
        assert metrics.max_reconstructed_logp_error <= 1e-5
        assert math.isfinite(metrics.calibration)
    probabilities = probability()
    assert probabilities[0] > before + .1
    assert probabilities.sum().item() == pytest.approx(1.)
    assert torch.isfinite(probabilities).all() and (probabilities > 0).all()
    assert not torch.equal(gru_before, model.event_gru.weight_ih)
    assert pool.read_bytes() == original


def test_rollout_logps_reconstruct_before_update(tiny, tmp_path):
    from brass_rl.ppo import update_policy
    model, components = tiny
    path = tmp_path/'initial.pt'
    torch.save(model.state_dict(), path)
    frozen_bytes = path.read_bytes()
    streams = SeedStreams(9123)
    league = League(streams, Snapshot('initial',path,0,'initial'), LeagueConfig((0,0,1)))
    config = TrainConfig(model=model.config, games_per_update=1, concurrent_games=1)
    episodes = collect_batch(model, league, streams, 0, 1, 1, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = update_policy(model, optimizer, episodes, replace(config.ppo, epochs=1), random.Random(1))
    assert metrics.max_reconstructed_logp_error <= 1e-5
    assert metrics.value_decisions == len(episodes[0].decisions)
    assert metrics.policy_decisions == sum(d.observation.num_legal()>1 for d in episodes[0].decisions)
    assert path.read_bytes() == frozen_bytes


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_kl_stop_and_nonfinite_abort(tiny):
    from brass_rl.ppo import update_policy
    model, components = tiny
    config = PPOConfig(epochs=4, target_kl=.03)
    clean_model = deepcopy(model.state_dict())
    class JumpAdam(torch.optim.Adam):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            with torch.no_grad():
                model.candidate_score[-1].weight.mul_(10000)
            return result
    optimizer = JumpAdam(model.parameters(), lr=.01)
    metrics = update_policy(model, optimizer, toy_episodes(model, components), config, random.Random(1))
    assert metrics.kl_stopped and metrics.optimizer_steps == 1
    assert metrics.approx_kl > .03
    # Prime real Adam state, then corrupt the second step of a fresh update.
    model.load_state_dict(clean_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    update_policy(model, optimizer, toy_episodes(model, components), replace(config, epochs=1), random.Random(1))
    before_model, before_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    episodes = toy_episodes(model, components)
    rng = random.Random(3)
    before_rng = rng.getstate()
    original_step = optimizer.step
    calls = []
    def corrupt_second_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            with torch.no_grad():
                next(model.parameters()).fill_(float('nan'))
        return result
    optimizer.step = corrupt_second_step
    with pytest.raises(ValueError, match='finite|rolled back'):
        update_policy(model, optimizer, episodes, config, rng)
    assert len(calls) == 2
    assert_nested_equal(model.state_dict(), before_model)
    assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert rng.getstate() == before_rng
    assert all(p.grad is None for p in model.parameters())


def test_frozen_weights_never_receive_gradients_and_old_mismatch_rejects(tiny):
    from brass_rl.ppo import update_policy
    model, components = tiny
    episodes = toy_episodes(model, components)
    optimizer = torch.optim.Adam(model.parameters())
    before = deepcopy(model.state_dict())
    bad = replace(episodes[0], decisions=(replace(episodes[0].decisions[0], old_log_prob=-99.),))
    with pytest.raises(ValueError, match='reconstruct|rollout'):
        update_policy(model, optimizer, [bad]+episodes[1:], PPOConfig(), random.Random(1))
    assert_nested_equal(model.state_dict(), before)
    frozen = replace(episodes[0], lineup=Lineup((LineupEntry('frozen','old'), LineupEntry('current', trainable=True),
                                                   LineupEntry('random'), LineupEntry('greedy'))))
    with pytest.raises(ValueError, match='current'):
        update_policy(model, optimizer, [frozen], PPOConfig(), random.Random(1))
    assert all(p.grad is None for p in model.parameters())


def test_burnin_has_no_gradient_and_main_events_recompute(tiny):
    from brass_rl.ppo import _sequence_stats
    from brass_rl.sequences import sequence_batches
    model, components = tiny
    ep = toy_episodes(model, components)[0]
    targets = episode_targets(ep)
    batch = next(sequence_batches([ep], targets, 1, 16, random.Random(1)))
    new = _sequence_stats(model, batch, components)
    assert new.log_prob.item() == pytest.approx(ep.decisions[0].old_log_prob, abs=1e-6)
    new.values.sum().backward()
    assert model.event_gru.weight_ih.grad is None  # only event1 was burn-in
    assert model.value_head.weight.grad is not None
    model.zero_grad(set_to_none=True)
    full = next(sequence_batches([ep], targets, 64, 16, random.Random(1)))
    new = _sequence_stats(model, full, components)
    new.values.sum().backward()
    assert model.event_gru.weight_ih.grad.abs().sum() > 0
    baseline = new.values.detach().clone()
    with torch.no_grad():
        model.event_gru.bias_ih.add_(1.)
    fresh = _sequence_stats(model, full, components)
    assert not torch.allclose(baseline, fresh.values)  # no cached learned embeddings/memory


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_gradient_aborts_and_restores_numpy_rng(tiny, bad):
    from brass_rl.ppo import update_policy
    model, components = tiny
    episodes = toy_episodes(model, components)
    optimizer = torch.optim.Adam(model.parameters())
    before = deepcopy(model.state_dict())
    rng = np.random.default_rng(1)
    state = deepcopy(rng.bit_generator.state)
    handle = model.value_head.weight.register_hook(lambda grad: grad*bad)
    with pytest.raises(ValueError, match='gradients.*finite'):
        update_policy(model, optimizer, episodes, PPOConfig(), rng)
    handle.remove()
    assert_nested_equal(model.state_dict(), before)
    assert not optimizer.state
    assert rng.bit_generator.state == state
    assert all(p.grad is None for p in model.parameters())


def test_empty_rollout_is_finite(tiny):
    from brass_rl.ppo import update_policy
    model, components = tiny
    optimizer = torch.optim.Adam(model.parameters())
    metrics = update_policy(model, optimizer, [], PPOConfig(), random.Random(1))
    assert metrics.optimizer_steps == metrics.value_decisions == 0
    assert metrics.total == 0.


def singleton_episode(model, components):
    game = native_game({'Shortfall': {'player': 2, 'owed': 3}})
    obs = game.observation(2)
    assert obs.num_legal() == 1
    initial = obs.public_json()
    h0 = model.initial_memory(1, model.device)
    with torch.no_grad():
        old = policy_stats(model([encode_observation(obs, components)], h0), torch.tensor([0]))
    decision = Decision(2, 0, 2, obs, 0, old.log_prob.item(), tuple(old.values[0].tolist()))
    game.apply_index(0)
    event = PublicEvent(1, 2, obs.legal_moves()[0], game.observation(2).public_json())
    with torch.no_grad():
        h1 = model.update_memory([encode_public(event, 2, initial, components)], h0)
    lineup = Lineup(tuple(LineupEntry('current', trainable=True) if s == 2 else LineupEntry('random')
                         for s in range(4)))
    return Episode(2, initial, (event,), {2: (h0, h1)}, (decision,), lineup,
                   GameResult(1, ('a','b','c','d'), (0,1,2,3), (4,3,2,1), 1))


def test_singleton_oldlog_is_validated_and_value_is_trained(tiny):
    from brass_rl.ppo import update_policy
    model, components = tiny
    ep = singleton_episode(model, components)
    optimizer = torch.optim.Adam(model.parameters())
    bad = replace(ep, decisions=(replace(ep.decisions[0], old_log_prob=.1),))
    with pytest.raises(ValueError, match='reconstruction'):
        update_policy(model, optimizer, [bad], PPOConfig(), random.Random(1))
    assert not optimizer.state
    before = model.value_head.weight.detach().clone()
    metrics = update_policy(model, optimizer, [ep], PPOConfig(epochs=1), random.Random(1))
    assert metrics.optimizer_steps == metrics.value_decisions == 1
    assert metrics.policy_decisions == 0
    assert metrics.entropy == metrics.policy == metrics.approx_kl == metrics.clip_fraction == 0.
    assert metrics.value > 0
    assert not torch.equal(before, model.value_head.weight)


def test_gradient_norm_is_clipped_before_optimizer_step(tiny):
    from brass_rl.ppo import update_policy
    model, components = tiny
    class CheckedAdam(torch.optim.Adam):
        def step(self, *args, **kwargs):
            norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
            assert norm <= .50001
            return super().step(*args, **kwargs)
    optimizer = CheckedAdam(model.parameters(), lr=.0003)
    metrics = update_policy(model, optimizer, toy_episodes(model, components),
                            PPOConfig(epochs=1, value_coef=100.), random.Random(1))
    assert metrics.gradient_norm > .5  # proves that clipping actually reduced the norm


def test_finite_ratios_with_kl_reduction_overflow_reject_loss():
    from brass_rl.ppo import ppo_loss
    # exp(87.9) is individually finite in float32 (~1.494e38), but its
    # four-row reduction overflows. The objective is independently exactly zero.
    new = PolicyStats(torch.full((4,), -.1), torch.zeros(4), torch.zeros((4, 4)))
    assert torch.isfinite((new.log_prob + 88.).exp()).all()
    with pytest.raises(ValueError, match='approx_kl.*finite'):
        ppo_loss(new, [-88.]*4, [0.]*4, [[0.]*4]*4, [True]*4, PPOConfig())


def test_post_step_nonfinite_kl_aborts_with_complete_rollback(tiny, monkeypatch):
    import brass_rl.ppo as ppo
    model, components = tiny
    config = PPOConfig(epochs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # Existing Adam moments/step counters must survive rollback, too.
    ppo.update_policy(model, optimizer, toy_episodes(model, components), config, random.Random(0))
    episodes = toy_episodes(model, components)
    before_model, before_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    rng = random.Random(4)
    before_rng = rng.getstate()
    real_loss, real_step = ppo.ppo_loss, optimizer.step
    completed_steps = []
    def step(*args, **kwargs):
        result = real_step(*args, **kwargs)
        completed_steps.append(1)
        return result
    def inject_post_step_kl(*args, **kwargs):
        terms = real_loss(*args, **kwargs)
        if not torch.is_grad_enabled():
            # Inject only into the post-step diagnostic, after a real Adam step.
            assert completed_steps
            return replace(terms, approx_kl=terms.approx_kl.new_tensor(float('inf')))
        return terms
    monkeypatch.setattr(optimizer, 'step', step)
    monkeypatch.setattr(ppo, 'ppo_loss', inject_post_step_kl)
    with pytest.raises(ValueError, match='rolled back.*approx_kl.*finite'):
        ppo.update_policy(model, optimizer, episodes, config, rng)
    assert completed_steps == [1]
    assert_nested_equal(model.state_dict(), before_model)
    assert_nested_equal(optimizer.state_dict(), before_optimizer)
    assert rng.getstate() == before_rng
    assert all(p.grad is None for p in model.parameters())
    assert model.training
