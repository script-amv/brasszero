# SPDX-License-Identifier: Apache-2.0
"""Small native observations exercise the candidate's recurrent optimization."""
from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brass_rl.config import PPOConfig
from brass_rl.features import encode_observation, encode_public
from brass_rl.ppo import _loss, _prepare_sequence
from brass_rl.returns import episode_targets, score_outcomes
from brass_rl.sequences import sequence_batches
from test_ppo import singleton_episode, tiny, toy_episodes  # noqa: F401
from test_train import assert_nested_equal


BC = PPOConfig(reward='score', critic_target='terminal')


def eight(model, components):
    pair = toy_episodes(model, components)
    result = []
    for i in range(8):
        ep = pair[i % 2]
        result.append(replace(ep, game_id=i, decisions=(replace(ep.decisions[0], game_id=i),),
                              result=replace(ep.result, vp=(100-i*10, 12, 44, 80))))
    return result


def test_distinct_games_in_full_groups_and_temporal_monitor():
    from brass_rl.stability import distinct_groups, monitor_sample
    fragments = [SimpleNamespace(batch=SimpleNamespace(game_id=g, start_ply=p, viewer=0),
                                 eligible=1) for g in range(8) for p in (0, 64, 128)]
    random.Random(2).shuffle(fragments)
    groups = list(distinct_groups(fragments))
    assert sorted(id(f) for group in groups for f in group) == sorted(map(id, fragments))
    assert len(groups) == 6
    assert all(len({f.batch.game_id for f in group}) == 4 for group in groups)
    sample = monitor_sample(fragments)
    assert len(sample) == len({f.batch.game_id for f in sample}) == 8
    assert {f.batch.start_ply for f in sample} == {0, 64, 128}
    assert sample == monitor_sample(list(reversed(fragments)))


def test_uneven_game_counts_leave_distinct_smaller_tail_groups():
    from brass_rl.stability import distinct_groups
    items = [SimpleNamespace(batch=SimpleNamespace(game_id=g)) for g in [0,0,0,0,1,2,3]]
    groups = [[f.batch.game_id for f in group] for group in distinct_groups(items)]
    assert all(len(group) == len(set(group)) for group in groups)
    assert sorted(map(len, groups)) == [1,1,1,4]


def test_critic_values_and_boundaries_replay_its_own_public_history(tiny):
    from brass_rl.stability import critic_rollout
    actor, components = tiny
    ep = toy_episodes(actor, components)[0]
    c = deepcopy(actor)
    obs = encode_observation(ep.decisions[0].observation, components)
    with torch.no_grad():
        assert torch.equal(c.values_only([obs], ep.boundary_states[0][1]),
                           actor([obs], ep.boundary_states[0][1]).values)
        c.event_gru.bias_ih.add_(1.)
        own = c.update_memory([encode_public(ep.events[0], 0, ep.initial_public_json, components)],
                              c.initial_memory(1, c.device))
        want = c([obs], own).values
    fake = replace(ep, boundary_states={0: tuple(torch.ones_like(own)*99 for _ in range(3))})
    projected = critic_rollout(c, fake, components)
    assert torch.equal(projected.boundary_states[0][1], own)
    assert not torch.equal(own, ep.boundary_states[0][1])
    np.testing.assert_array_equal(projected.decisions[0].old_values, want.detach().tolist()[0])
    assert projected.decisions[0].old_log_prob == ep.decisions[0].old_log_prob
    assert not projected.boundary_states[0][1].requires_grad


def test_own_seat_gae_uses_frozen_critic_values_and_its_boundary_at_nonzero_ply(tiny, monkeypatch):
    import brass_rl.stability as treatment
    from brass_rl.distributions import policy_stats
    from helpers import shortfall_game

    actor, components = tiny
    critic = deepcopy(actor)
    with torch.no_grad():
        critic.event_gru.bias_ih.add_(1.)
    cfg = replace(BC, sequence_events=1, burn_in_events=0)
    episodes = []
    raw = {}
    expected_starts = {}
    for ep in eight(actor, components):
        d = ep.decisions[0]
        first_obs = shortfall_game().observation(0)
        with torch.no_grad():
            obs = encode_observation(first_obs, components)
            state = actor.initial_memory(1, actor.device)
            old = policy_stats(actor([obs], state), torch.tensor([0])).log_prob.item()
            v0 = critic.values_only([obs], state)[0, 0].item()
            h1 = critic.update_memory([encode_public(ep.events[0], 0, ep.initial_public_json, components)], state)
            v1 = critic.values_only([encode_observation(d.observation, components)], h1)[0, 0].item()
        expected_starts[ep.game_id] = h1
        # Poison only the shared rollout value estimates. The candidate must
        # reconstruct its own values before computing GAE or training its critic.
        d0 = replace(d, ply=0, observation=first_obs, index=0, old_log_prob=old, old_values=(float('nan'),)*4)
        d1 = replace(d, old_values=(float('nan'),)*4)
        episodes.append(replace(ep, decisions=(d0, d1)))
        reward = score_outcomes(ep.result.vp)[0]
        raw[d1.key] = reward-v1
        if first_obs.num_legal() > 1:
            raw[d0.key] = v1-v0+.95*(reward-v1)
    mean, std = np.mean(list(raw.values())), np.std(list(raw.values()))
    expected = {key: (v-mean)/std for key,v in raw.items()}
    seen = []
    actor_group, critic_group = treatment.actor_group, treatment.critic_group
    def check_actor(model, adam, group, comp, config):
        for f in group:
            for d, target in zip(f.batch.decisions, f.batch.targets):
                assert target.self_advantage == pytest.approx(expected.get(d.key, 0.), abs=2e-5)
                seen.append(d.key)
        return actor_group(model, adam, group, comp, config)
    def check_critic(model, adam, group, comp, config):
        for f in group:
            if f.batch.start_ply == 1:
                assert torch.equal(f.critic_start, expected_starts[f.batch.game_id])
                assert not torch.equal(f.critic_start, f.batch.recorded_start_state)
        return critic_group(model, adam, group, comp, config)
    monkeypatch.setattr(treatment, 'actor_group', check_actor)
    monkeypatch.setattr(treatment, 'critic_group', check_critic)
    result = treatment.update_candidate(actor, torch.optim.Adam(actor.parameters(), lr=.0003),
        critic, torch.optim.Adam(critic.parameters(), lr=.0003), episodes, cfg, random.Random(4))
    assert set(seen) >= set(expected)
    assert result['critic_value_visits'] == 4*16


def test_decision_weighted_actor_and_critic_gradients(tiny):
    from brass_rl.stability import Fragment, actor_group, critic_group
    actor, components = tiny
    episodes = toy_episodes(actor, components)
    # A three-decision fragment and a one-decision fragment. Independent
    # gradient oracle sums each row, including the observer-relative value MSE.
    batches = []
    for i, ep in enumerate(episodes):
        batch = next(sequence_batches([ep], episode_targets(ep, reward='score', critic_target='terminal'),
                                      64, 16, random.Random(0)))
        decisions = batch.decisions
        if i == 0:
            # Two eligible moves plus one forced singleton share this fragment.
            d = decisions[0]
            forced = singleton_episode(actor, components).decisions[0]
            decisions = (d, replace(d, ply=2, observation=forced.observation, index=0,
                                    old_log_prob=0.), replace(d, ply=3))
            batch = replace(batch, main_events=(batch.main_events[0],)+(batch.main_events[1],)*3,
                            end_ply=4, decisions=decisions, decision_positions=(1, 2, 3),
                            targets=(batch.targets[0], replace(batch.targets[0], policy_mask=False), batch.targets[0]))
        batch = replace(batch, targets=tuple(replace(t, self_advantage=(1., -2.)[i]) for t in batch.targets))
        prepared = _prepare_sequence(batch, components)
        batches.append(Fragment(batch, prepared, batch.recorded_start_state, sum(t.policy_mask for t in batch.targets)))
    critic = deepcopy(actor)
    cfg = replace(BC, max_grad_norm=1e6)
    for model, group_fn, objective in ((actor, actor_group, 'actor'), (critic, critic_group, 'critic')):
        reference = deepcopy(model)
        for f in batches:
            _, terms = _loss(reference, f.batch, components, cfg, f.prepared)
            loss = terms.policy-cfg.entropy_coef*terms.entropy if objective == 'actor' else cfg.value_coef*terms.value
            weight = f.eligible/3 if objective == 'actor' else len(f.batch.decisions)/4
            (loss * weight).backward()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        captured = {}
        original = opt.step
        def step():
            captured.update({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
            return original()
        opt.step = step
        group_fn(model, opt, batches, components, cfg)
        expected = {k: p.grad for k, p in reference.named_parameters() if p.grad is not None}
        assert captured.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(captured[name], expected[name], atol=2e-7, rtol=1e-5)


def test_cross_game_kl_stop_and_independent_four_epoch_critic(tiny):
    from brass_rl.stability import update_candidate
    model, components = tiny
    episodes = eight(model, components)
    critic = deepcopy(model)
    critic_before = deepcopy(critic.state_dict())
    actor_at_stop = {}
    class JumpAdam(torch.optim.Adam):
        def step(self):
            assert_nested_equal(critic.state_dict(), critic_before)
            super().step()
            with torch.no_grad():
                model.candidate_score[-1].weight.mul_(10000)
            actor_at_stop.update(deepcopy(model.state_dict()))
    opt = JumpAdam(model.parameters(), lr=.0003)
    copt = torch.optim.Adam(critic.parameters(), lr=.0003)
    result = update_candidate(model, opt, critic, copt, episodes, BC, random.Random(9))
    assert result['actor_stop_reason'] == 'representative_kl'
    assert result['kl_stopped'] and result['actor_steps'] == result['kl_checks'] == 1
    assert result['monitor_games'] == result['monitor_policy_decisions'] == 8
    assert result['actor_policy_visits'] == result['actor_fragment_visits'] == 4
    assert result['actor_four_game_steps'] == 1
    assert result['critic_fragment_visits'] == 32 and result['critic_four_game_steps'] == 8
    assert result['critic_steps'] == 8 and result['critic_value_visits'] == 32
    assert result['final_policy_decisions'] == 8
    old = torch.tensor([ep.decisions[0].old_log_prob for ep in episodes])
    new = torch.tensor([ep.decisions[0].old_log_prob for ep in eight(model, components)])
    ratio = new-old
    assert result['final_kl'] == pytest.approx((ratio.expm1()-ratio).mean().item(), rel=1e-5)
    assert result['final_clip_fraction'] == pytest.approx(((ratio.exp()-1).abs() > .2).float().mean().item())
    assert_nested_equal(model.state_dict(), actor_at_stop)
    assert not torch.equal(critic_before['value_head.weight'], critic.value_head.weight)


def test_guard_stops_on_moves_outside_the_just_trained_games(tiny):
    from brass_rl.stability import update_candidate
    actor, components = tiny
    # Uniform behavior makes the two action probabilities exactly .5. Four
    # trained games selected action 0, four other monitored games selected 1.
    original_weight = actor.candidate_score[-1].weight.detach().clone()
    with torch.no_grad():
        actor.candidate_score[-1].weight.zero_()
    pair = toy_episodes(actor, components)
    episodes = []
    for game in range(8):
        ep = pair[int(game < 4)]
        episodes.append(replace(ep, game_id=game, decisions=(replace(ep.decisions[0], game_id=game),)))
    critic = deepcopy(actor)
    class NoShuffle(random.Random):
        def shuffle(self, items):
            pass
    class ShiftProbability(torch.optim.Adam):
        def step(self):
            super().step()
            with torch.no_grad():
                obs = encode_observation(episodes[0].decisions[0].observation, components)
                h = actor.update_memory([encode_public(episodes[0].events[0], 0,
                    episodes[0].initial_public_json, components)], actor.initial_memory(1, actor.device))
                actor.candidate_score[-1].weight.copy_(original_weight)
                logits = actor([obs], h).logits
                scale = np.log(5/3)/(logits[0]-logits[1]).item()
                actor.candidate_score[-1].weight.mul_(scale)
    result = update_candidate(actor, ShiftProbability(actor.parameters()), critic,
                              torch.optim.Adam(critic.parameters()), episodes, BC, NoShuffle(0))
    # KL of ratio 1.25 stays below threshold for the four action-0 games;
    # ratio .75 on untrained games pushes the fixed cross-game average over.
    assert 1.25-1-np.log(1.25) < .03
    assert result['monitor_kl'] == pytest.approx(-.5*np.log(1.25*.75), abs=1e-5)
    assert result['monitor_kl'] > .03
    assert result['actor_steps'] == 1 and result['actor_stop_reason'] == 'representative_kl'
    assert result['kl_check_interval_actor_steps'] == 1


def test_singletons_train_only_critic_on_relative_four_vector(tiny):
    from brass_rl.stability import update_candidate
    model, components = tiny
    ep = singleton_episode(model, components)
    critic = deepcopy(model)
    original = deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters())
    copt = torch.optim.Adam(critic.parameters())
    result = update_candidate(model, opt, critic, copt, [ep], BC, random.Random(1))
    assert result['actor_steps'] == result['final_policy_decisions'] == 0
    assert result['critic_steps'] == 4
    expected = np.roll(score_outcomes(ep.result.vp), -2)
    assert result['critic_mse_before'] == pytest.approx(np.mean((expected-ep.decisions[0].old_values)**2))
    assert result['critic_mse_after'] < result['critic_mse_before']
    assert_nested_equal(model.state_dict(), original)
    assert not opt.state and copt.state


def test_critic_failure_rolls_back_both_networks_adams_and_rng(tiny):
    from brass_rl.artifacts import capture_rng_states
    from brass_rl.stability import update_candidate
    model, components = tiny
    critic = deepcopy(model)
    opt = torch.optim.Adam(model.parameters())
    copt = torch.optim.Adam(critic.parameters())
    rng = random.Random(3)
    update_candidate(model, opt, critic, copt, eight(model, components), BC, rng)
    episodes = eight(model, components)
    before = deepcopy([m.state_dict() for m in (model, opt, critic, copt)])
    global_rng, shuffle = capture_rng_states(), rng.getstate()
    original = copt.step
    def fail():
        original()
        random.random()
        np.random.rand()
        torch.rand(1)
        raise ArithmeticError('injected after critic Adam')
    copt.step = fail
    with pytest.raises(ValueError, match='rolled back.*injected'):
        update_candidate(model, opt, critic, copt, episodes, BC, rng)
    assert_nested_equal(before, [m.state_dict() for m in (model, opt, critic, copt)])
    assert_nested_equal(global_rng, capture_rng_states())
    assert shuffle == rng.getstate()
    assert all(p.grad is None for m in (model, critic) for p in m.parameters())


def test_full_epoch_guard_catches_drift_outside_fixed_sample(tiny):
    from brass_rl.stability import update_candidate
    actor, components = tiny
    weight = actor.candidate_score[-1].weight
    original = weight.detach().clone()
    with torch.no_grad():
        weight.zero_()
        for p in actor.event_gru.parameters():
            p.zero_()  # Identical logit shift at both temporal positions.
    template = toy_episodes(actor, components)[0]
    episodes = []
    for game in range(8):
        d = template.decisions[0]
        # Probe samples the early fragment in the first four games and the late
        # fragment in the other four. Those eight moves select action zero.
        decisions = tuple(replace(d, game_id=game, ply=ply, index=int(ply != int(game >= 4))) for ply in (0,1))
        episodes.append(replace(template, game_id=game, decisions=decisions))
    critic = deepcopy(actor)
    obs = encode_observation(template.decisions[0].observation, components)
    with torch.no_grad():
        weight.copy_(original)
        logits = actor([obs], actor.initial_memory(1, actor.device)).logits
        shifted = original*(np.log(5/3)/(logits[0]-logits[1]).item())
        weight.zero_()

    class ShiftActor(torch.optim.Adam):
        def step(self):
            with torch.no_grad():
                weight.copy_(shifted)

    cfg = replace(BC, sequence_events=1, burn_in_events=0)
    result = update_candidate(actor, ShiftActor(actor.parameters()), critic, torch.optim.Adam(critic.parameters()),
                              episodes, cfg, random.Random(13))
    assert result['monitor_kl'] == pytest.approx(.25-np.log(1.25), abs=1e-5)
    assert result['final_kl'] == pytest.approx(-.5*np.log(1.25*.75), abs=1e-5)
    assert result['monitor_kl'] < .03 < result['final_kl']
    assert result['actor_stop_reason'] == 'full_epoch_kl' and result['kl_stopped']
    assert result['actor_steps'] == 4 and result['critic_steps'] == 16
    assert result['full_epoch_kl_checks'] == 1 and result['full_epoch_policy_decision_visits'] == 16
    assert result['last_full_epoch_kl'] == result['final_kl']


def test_probe_stop_on_final_group_counts_completed_full_epoch(tiny):
    from brass_rl.stability import update_candidate
    actor, components = tiny
    episodes = eight(actor, components)[:4]
    critic = deepcopy(actor)
    class Jump(torch.optim.Adam):
        def step(self):
            super().step()
            with torch.no_grad():
                actor.candidate_score[-1].weight.mul_(10000)
    result = update_candidate(actor, Jump(actor.parameters(),lr=.0003), critic, torch.optim.Adam(critic.parameters()),
                              episodes, BC, random.Random(9))
    assert result['actor_stop_reason'] == 'representative_kl' and result['actor_steps'] == 1
    assert result['full_epoch_kl_checks'] == 1
    assert result['full_epoch_policy_decision_visits'] == result['final_policy_decisions'] == 4
    assert result['last_full_epoch_kl'] == result['final_kl']
