# SPDX-License-Identifier: Apache-2.0
"""Native information and public-memory lifecycle; no new complete games here."""
from dataclasses import replace
import json
import random

import brass
import numpy as np
import pytest
import torch

from brass_ai.arena import RefereeGame
from brass_rl.agent import TorchAgent
from brass_rl.config import ModelConfig
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData
from brass_search import SearchAgent, SearchConfig
from brass_search.identity import search_identity, player_manifest


@pytest.fixture(scope='module')
def policy():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20)
        components = ComponentData.load()
        model = BrassPolicy(ModelConfig(), components.schema, components).eval().requires_grad_(False)
    return model, components


def new_agent(policy, config=None):
    return SearchAgent(*policy, 81, config or SearchConfig(root_candidates=2, worlds_per_round=2, rollout_depth=2, max_rollouts=4), base_policy_id='policy-sha256-v1:' + 'a'*64)


def test_zero_matches_baseline_on_shared_actual_history(policy):
    agent = new_agent(policy, SearchConfig(max_rollouts=0))
    baseline = TorchAgent(*policy, 81)
    referee = RefereeGame(52, [agent] + [brass.RandomBot(i) for i in range(1, 4)])
    baseline.reset(0, agent.latest_public_json)
    decisions = 0
    for _ in range(36):
        seat, obs = referee.pending()
        if seat == 0:
            index = agent.act(obs)
            assert index == baseline.act(obs)
            assert torch.equal(agent._generator.get_state(), baseline._generator.get_state())
            decisions += 1
            assert agent.last_diagnostics['reason'] == 'zero_budget'
        else:
            index = 0
        event = referee.apply_index(index)
        baseline.observe_public(event)
        assert torch.equal(agent.memory_snapshot(), baseline.memory_snapshot())
    assert decisions >= 4


def test_search_preserves_baseline_rng_observation_and_four_memories(policy):
    agent = new_agent(policy)
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    agent.reset(obs.me, obs.public_json())
    baseline = TorchAgent(*policy, 81)
    baseline.reset(obs.me, obs.public_json())
    memories = agent.public_memory_snapshots()
    before = (obs.to_json(), game.to_json(), torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state())
    index = agent.act(obs)
    baseline.act(obs)
    assert torch.equal(agent._generator.get_state(), baseline._generator.get_state())
    assert all(torch.equal(a, b) for a, b in zip(memories, agent.public_memory_snapshots()))
    assert before[0] == obs.to_json() and before[1] == game.to_json()
    assert torch.equal(before[2], torch.random.get_rng_state()) and before[3] == random.getstate()
    assert np.array_equal(before[4][1], np.random.get_state()[1])
    assert 0 <= index < obs.num_legal()
    d = agent.last_diagnostics
    assert d['rollouts'] == 4 and d['native_applies'] <= 8 and d['completed_rounds'] == 1
    assert d['max_live_worlds'] == 2 and d['max_live_continuations'] == 1
    assert d['root_candidates_encoded'] == obs.num_legal()
    assert d['inference_calls'] >= 1 and d['rollout_candidates_encoded'] > 0


def test_lifecycle_and_chosen_search_move(policy):
    agent = new_agent(policy)
    referee = RefereeGame(52, [brass.RandomBot(i) for i in range(4)])
    seat, obs = referee.pending()
    with pytest.raises(RuntimeError, match='reset'):
        agent.act(obs)
    agent.reset((seat + 1) % 4, obs.public_json())
    with pytest.raises(ValueError, match='viewer|seat'):
        agent.act(obs)
    agent.reset(seat, obs.public_json())
    index = agent.act(obs)
    with pytest.raises(RuntimeError, match='awaiting'):
        agent.act(obs)
    event = referee.apply_index(index)
    original = agent.public_memory_snapshots()
    for bad in (replace(event, ply=2), replace(event, actor=(seat+1)%4), replace(event, move_json='"EndAction"')):
        with pytest.raises(ValueError):
            agent.observe_public(bad)
        assert all(torch.equal(a, b) for a, b in zip(original, agent.public_memory_snapshots()))
    agent.observe_public(event)
    for viewer, memory in enumerate(agent.public_memory_snapshots()):
        reference = TorchAgent(*policy, 81)
        reference.reset(viewer, obs.public_json())
        reference.observe_public(event)
        assert torch.equal(memory, reference.memory_snapshot())
    with pytest.raises(ValueError):
        agent.observe_public(event)


def test_hidden_world_and_hand_order_invariance(policy):
    game = brass.Game(77)
    original = game.observation(game.current_player())
    worlds = [original.determinize(seed) for seed in (99, 100)]
    assert worlds[0].to_json() != worlds[1].to_json()
    results = []
    for world in worlds:
        obs = world.observation(original.me)
        for reverse in (False, True):
            if reverse:
                data = json.loads(world.to_json())
                data['players'][original.me]['hand'].reverse()
                reordered = brass.Game.from_json(json.dumps(data))
                obs = reordered.observation(original.me)
            agent = new_agent(policy)
            agent.reset(obs.me, obs.public_json())
            index = agent.act(obs)
            d = {k: v for k, v in agent.last_diagnostics.items() if k not in ('elapsed_seconds', 'wall_overrun_seconds')}
            results.append((json.loads(obs.legal_moves()[index]), d))
    assert all(result == results[0] for result in results)


def test_identity_config_sources_and_seed_independence(policy):
    a = new_agent(policy)
    b = SearchAgent(*policy, 90, a.config, base_policy_id=a.base_policy_id)
    assert a.replay_identity == b.replay_identity
    assert a.replay_identity.startswith('search-sha256-v1:')
    assert a.replay_identity != search_identity(a.base_policy_id, policy[1].schema.fingerprint, replace(a.config, rollout_depth=3))
    assert a.replay_identity != search_identity('other-policy', policy[1].schema.fingerprint, a.config)


def test_wall_cap_fallback_and_nonfinite_failure(policy):
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    agent = new_agent(policy, SearchConfig(wall_time_seconds=1e-12))
    reference = TorchAgent(*policy, 81)
    for a in (agent, reference):
        a.reset(obs.me, obs.public_json())
    assert agent.act(obs) == reference.act(obs)
    assert agent.last_diagnostics['fallback'] and agent.last_diagnostics['reason'] == 'wall_cap'
    assert agent.last_diagnostics['wall_overrun_seconds'] > 0
    # Actual malformed model output must fail instead of invoking a policy fallback.
    model, components = policy
    with torch.no_grad():
        saved = model.value_head.bias[0].clone()
        model.value_head.bias[0] = float('nan')
    try:
        broken = new_agent(policy)
        broken.reset(obs.me, obs.public_json())
        with pytest.raises(ValueError, match='finite'):
            broken.act(obs)
    finally:
        with torch.no_grad():
            model.value_head.bias[0] = saved


def test_loaded_policy_manifest_and_registered_private_seats(policy, tmp_path):
    from brass_search import load_search_agent, register_search_checkpoint
    from brass_ai.agents import make_agent
    from brass_rl.policy_io import load_policy
    from brass_rl.artifacts import export_model_policy
    path = tmp_path / 'qa-policy'
    export_model_policy(policy[0], 600, path)
    cfg = SearchConfig(max_rollouts=0)
    before = torch.random.get_rng_state().clone()
    a = load_search_agent(path, 1, cfg)
    register_search_checkpoint('search-test', path, cfg)
    b, c = make_agent('search-test', 2), make_agent('search-test', 3)
    assert a.replay_identity == b.replay_identity == c.replay_identity
    assert b.model is c.model and b._generator is not c._generator
    assert a.base_update == 600 and a.manifest['config']['max_rollouts'] == 0
    assert 'path' not in json.dumps(a.manifest) and 'seed' not in a.manifest
    metadata = load_policy(path)[2]
    plain = player_manifest(metadata)
    assert plain['id'] == a.base_policy_id and plain['mode'] == 'sample'
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError, match='config'):
        load_search_agent('/not/a/policy', 1, {})


def test_native_retained_positions_both_eras_followups_and_boundaries(policy):
    from pathlib import Path
    from brass_search.simulator import SampledSimulator, normalized_observation
    from brass_search.inference import move_key
    from brass_search.allocation import terminal_values
    # Native replays of retained QA fixtures: no new complete game or strength run.
    fixtures = Path(__file__).resolve().parents[3] / 'engine/tools/tests/fixtures/strategy'
    selected = {0: {19, 50, 67, 70, 71, 72, 146, 173}, 1: {78, 85, 155}}
    coverage = set()
    terminal_rollouts = 0
    for number, indices in selected.items():
        record = json.loads((fixtures / ('record-%d.json' % number)).read_text())
        game = brass.Game(record['initial']['seed'])
        for ply, item in enumerate(record['moves']):
            if ply in indices:
                actor = game.current_player()
                obs = game.observation(actor)
                public = json.loads(obs.public_json())
                phase = public['phase']
                coverage.add((public['era'], 'Shortfall' if isinstance(phase, dict) else phase))
                config = SearchConfig(root_candidates=2, worlds_per_round=1, rollout_depth=1, max_rollouts=2)
                agent = new_agent(policy, config)
                agent.reset(actor, obs.public_json())
                idx = agent.act(obs)
                assert 0 <= idx < obs.num_legal()
                if obs.num_legal() == 1:
                    assert agent.last_diagnostics['native_applies'] == 0
                    assert agent.last_diagnostics['reason'] == 'single_legal'
                for seed in (111, 222):
                    sampled = obs.determinize(seed)
                    assert normalized_observation(sampled.observation(actor)) == normalized_observation(obs)
                    sampled.apply(json.dumps(item['move']))
                # Force the retained boundary root through the actual simulator.
                counts = dict(native_applies=0, inference_calls=0, policy_forward_calls=0,
                              memory_update_calls=0, root_candidates_encoded=0, rollout_candidates_encoded=0,
                              max_live_worlds=0, max_live_continuations=0, worlds_sampled=0)
                simulator = SampledSimulator(policy[0], policy[1], obs, agent.public_memory_snapshots(),
                                             81, ply, config, counts, None)
                value = simulator(move_key(json.dumps(item['move'])), 0)
                assert 0 <= value <= 1 and counts['native_applies'] == 1
                if sampled.is_over():
                    terminal_rollouts += 1
                    assert value == terminal_values(sampled.ranking())[actor]
                with pytest.raises(ValueError, match='illegal|unknown variant'):
                    simulator('"invalid-move"', 1)
            game.apply(json.dumps(item['move']))
    assert {('canal', 'Shortfall'), ('rail', 'Shortfall'), ('canal', 'DevelopMore'),
            ('rail', 'SecondRail'), ('rail', 'SellMore'), ('rail', 'ChooseAction')} <= coverage
    assert terminal_rollouts == 2


def test_native_terminal_unique_first_tiebreaks():
    from brass_search.allocation import terminal_values
    # Strict native ranking includes income, then money. Synthetic QA fixtures only.
    state = json.loads(brass.Game(5).to_json())
    state['game_over'] = True
    for p in state['players']:
        p.update(vp=10, income_space=10, money=20)
    game = brass.Game.from_json(json.dumps(state))
    assert terminal_values(game.ranking()) == (0., 0., 0., 0.)
    state['players'][2]['money'] = 21
    game = brass.Game.from_json(json.dumps(state))
    assert terminal_values(game.ranking()) == (0., 0., 1., 0.)
    state['players'][1]['income_space'] = 20
    game = brass.Game.from_json(json.dumps(state))
    assert terminal_values(game.ranking()) == (0., 1., 0., 0.)


def test_each_real_event_updates_each_memory_once(policy, monkeypatch):
    calls = []
    model, components = policy
    original = model.update_memory
    def counted(events, memory):
        calls.append(events[0].actor)
        return original(events, memory)
    monkeypatch.setattr(model, 'update_memory', counted)
    agent = new_agent(policy)
    referee = RefereeGame(52, [brass.RandomBot(i) for i in range(4)])
    seat, obs = referee.pending()
    agent.reset(seat, obs.public_json())
    event = referee.apply_index(0)
    agent.observe_public(event)
    assert len(calls) == 4
    assert sorted(calls) == [0, 1, 2, 3]


def test_bonus_develop_native_actor_and_resource_followup(policy):
    from pathlib import Path
    import importlib.util
    path = Path(__file__).resolve().parents[3] / 'training/tests/helpers.py'
    spec = importlib.util.spec_from_file_location('search_qa_helpers', path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    game = helpers.native_game('BonusDevelop', rotation=2)
    actor = game.current_player()
    obs = game.observation(actor)
    agent = new_agent(policy)
    agent.reset(actor, obs.public_json())
    index = agent.act(obs)
    assert 0 <= index < obs.num_legal()
    assert agent.last_diagnostics['native_applies'] <= 8
    # Selected root validation uses native apply, even in this exceptional follow-up.
    game.apply(obs.legal_moves()[index])


def test_single_candidate_is_distinct_from_zero_budget(policy):
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    agent = new_agent(policy, SearchConfig(root_candidates=1))
    agent.reset(obs.me, obs.public_json())
    index = agent.act(obs)
    assert 0 <= index < obs.num_legal()
    assert agent.last_diagnostics['reason'] == 'single_candidate'
    assert not agent.last_diagnostics['fallback']
    assert agent.last_diagnostics['rollouts'] == agent.last_diagnostics['native_applies'] == 0


def test_pilot_native_work_ceiling(policy):
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    agent = new_agent(policy, SearchConfig())
    agent.reset(obs.me, obs.public_json())
    assert 0 <= agent.act(obs) < obs.num_legal()
    d = agent.last_diagnostics
    assert d['completed_rounds'] == 2 and d['rollouts'] == 12 and d['native_applies'] == 96
    assert d['policy_forward_calls'] == 97 and d['memory_update_calls'] == 384
    assert d['inference_calls'] == 481 and d['worlds_sampled'] == 4
    assert d['max_live_worlds'] == 2 and d['max_live_continuations'] == 1


def test_native_illegal_root_fails_without_policy_fallback(policy):
    from brass_search.simulator import SampledSimulator
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    agent = new_agent(policy)
    agent.reset(obs.me, obs.public_json())
    counters = dict(native_applies=0, inference_calls=0, policy_forward_calls=0,
                    memory_update_calls=0, root_candidates_encoded=0, rollout_candidates_encoded=0,
                    max_live_worlds=0, max_live_continuations=0, worlds_sampled=0)
    simulator = SampledSimulator(policy[0], policy[1], obs, agent.public_memory_snapshots(),
                                 81, 0, agent.config, counters, None)
    with pytest.raises(ValueError, match='illegal move'):
        simulator('"EndAction"', 0)
    assert counters['native_applies'] == 0


@pytest.mark.parametrize('max_rollouts,wall', [(0, None), (4, 1e-12)])
def test_exact_baseline_distribution_with_nondefault_chunk(policy, monkeypatch, max_rollouts, wall):
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    cfg = SearchConfig(candidate_chunk=1, max_rollouts=max_rollouts, wall_time_seconds=wall)
    search = new_agent(policy, cfg)
    baseline = TorchAgent(*policy, 81)
    for agent in (baseline, search):
        agent.reset(obs.me, obs.public_json())
    actual_sampler_inputs = []
    original_sample = torch.multinomial
    def sample(probabilities, *args, **kwargs):
        actual_sampler_inputs.append(probabilities.clone())
        return original_sample(probabilities, *args, **kwargs)
    monkeypatch.setattr(torch, 'multinomial', sample)
    forwards = []
    original_forward = policy[0].forward
    def forward(*args, **kwargs):
        forwards.append(kwargs.get('candidate_chunk', args[2] if len(args) > 2 else 1024))
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(policy[0], 'forward', forward)
    expected = baseline.act(obs)
    assert search.act(obs) == expected
    assert len(actual_sampler_inputs) == 2  # one draw each, no simulated draw at cap
    assert torch.equal(actual_sampler_inputs[0], actual_sampler_inputs[1])
    assert forwards == [1024, 1024]  # one complete root forward, reference batch shape


def test_root_only_nonfinite_event_is_transactional(policy, monkeypatch):
    agent = new_agent(policy, SearchConfig(max_rollouts=0))
    referee = RefereeGame(52, [brass.RandomBot(i) for i in range(4)])
    seat, obs = referee.pending()
    agent.reset(seat, obs.public_json())
    index = agent.act(obs)
    event = referee.apply_index(index)
    memories = agent.public_memory_snapshots()
    previous = (agent.ply, agent.latest_public_json, agent._awaiting)
    original = policy[0].update_memory
    calls = []
    def selective_failure(events, memory):
        calls.append(events[0].actor)
        result = original(events, memory)
        return torch.full_like(result, float('nan')) if events[0].actor == 0 else result
    monkeypatch.setattr(policy[0], 'update_memory', selective_failure)
    with pytest.raises(ValueError, match='finite'):
        agent.observe_public(event)
    assert (agent.ply, agent.latest_public_json, agent._awaiting) == previous
    assert all(torch.equal(a, b) for a, b in zip(memories, agent.public_memory_snapshots()))
    monkeypatch.setattr(policy[0], 'update_memory', original)
    agent.observe_public(event)
    assert agent.ply == 1 and agent._awaiting is None
    assert all(torch.isfinite(m).all() for m in agent.public_memory_snapshots())


@pytest.mark.parametrize('failure_stage', ['forward', 'simulation'])
def test_failed_act_restores_sampler_and_lifecycle_for_retry(policy, monkeypatch, failure_stage):
    from brass_search.simulator import SampledSimulator
    agent, reference = new_agent(policy), new_agent(policy)
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    for player in (agent, reference):
        player.reset(obs.me, obs.public_json())
    initial_rng = agent._generator.get_state().clone()
    initial_memories = agent.public_memory_snapshots()
    failure = ValueError('injected ' + failure_stage)
    def fail(*args, **kwargs):
        raise failure
    with monkeypatch.context() as patch:
        if failure_stage == 'forward':
            patch.setattr(policy[0], 'forward', fail)
        else:
            patch.setattr(SampledSimulator, '__call__', fail)
        with pytest.raises(ValueError) as exc:
            agent.act(obs)
        assert exc.value is failure
    assert torch.equal(initial_rng, agent._generator.get_state())
    assert agent._prepared is None and agent._awaiting is None
    assert agent.last_diagnostics == {}
    assert all(torch.equal(a, b) for a, b in zip(initial_memories, agent.public_memory_snapshots()))
    assert agent.act(obs) == reference.act(obs)
    assert torch.equal(agent._generator.get_state(), reference._generator.get_state())
    assert agent._awaiting == reference._awaiting


def terminal_prefix(policy):
    """Retained native prefix only: no newly played game, final decision not applied."""
    from pathlib import Path
    record = json.loads((Path(__file__).resolve().parents[3] /
                         'engine/tools/tests/fixtures/strategy/record-0.json').read_text())
    game = brass.Game(record['initial']['seed'])
    for item in record['moves'][:-1]:
        game.apply(json.dumps(item['move']))
    obs = game.observation(game.current_player())
    agent = new_agent(policy)
    agent.reset(obs.me, obs.public_json())
    return obs, agent.public_memory_snapshots(), record['moves'][-1]['move']


def simulation_counts():
    return dict(native_applies=0, inference_calls=0, policy_forward_calls=0,
                memory_update_calls=0, root_candidates_encoded=0, rollout_candidates_encoded=0,
                max_live_worlds=0, max_live_continuations=0, worlds_sampled=0)


def test_terminal_round_crossing_wall_cap_discards_all_partial_values(policy, monkeypatch):
    import time
    from brass_search.simulator import SampledSimulator
    from brass_search.inference import move_key
    from brass_search.allocation import Candidate, allocate
    obs, memories, move = terminal_prefix(policy)
    clock = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    original = policy[0].update_memory
    updates = []
    def crossing_update(events, memory):
        result = original(events, memory)
        updates.append(events[0].actor)
        if len(updates) == 8:  # final viewer of final continuation in the round
            clock[0] = 2.
        return result
    monkeypatch.setattr(policy[0], 'update_memory', crossing_update)
    cfg = SearchConfig(root_candidates=2, worlds_per_round=1, rollout_depth=1, max_rollouts=2)
    counts = simulation_counts()
    simulator = SampledSimulator(*policy, obs, memories, 81, 173, cfg, counts, 1.)
    # Allocate two synthetic comparison labels to the same genuine terminal root;
    # the test controls allocation only, keeping native terminal work/model calls real.
    result = allocate([Candidate('a', 0.), Candidate('b', 0.)], cfg,
                      lambda key, world: simulator(move_key(json.dumps(move)), world), 'fallback')
    assert result.selected == 'fallback' and result.completed_rounds == 0
    assert result.reason == 'wall_cap' and result.discarded_rollouts == result.rollouts == 2
    assert counts['native_applies'] == 2 and counts['memory_update_calls'] == 8
    assert result.means == {}


@pytest.mark.parametrize('stop_check', [2, 3])
def test_sampling_boundary_cap_counts_created_world(policy, monkeypatch, stop_check):
    from brass_search.simulator import SampledSimulator
    from brass_search.allocation import Candidate, allocate, BudgetStop
    import brass_search.simulator as simulator_module
    game = brass.Game(77)
    obs = game.observation(game.current_player())
    agent = new_agent(policy)
    agent.reset(obs.me, obs.public_json())
    calls = []
    def stop_after_determinize(deadline):
        calls.append(deadline)
        if len(calls) == stop_check:
            raise BudgetStop('wall_cap')
    monkeypatch.setattr(simulator_module, 'check_wall', stop_after_determinize)
    counts = simulation_counts()
    simulator = SampledSimulator(*policy, obs, agent.public_memory_snapshots(), 81, 0,
                                 agent.config, counts, 1.)
    result = allocate([Candidate('a', 0.), Candidate('b', 0.)], agent.config, simulator, 'fallback')
    assert result.reason == 'wall_cap' and result.rollouts == result.discarded_rollouts == 1
    assert counts['worlds_sampled'] == counts['max_live_worlds'] == 1
    assert counts['max_live_continuations'] == counts['native_applies'] == 0
