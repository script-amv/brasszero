# SPDX-License-Identifier: Apache-2.0
"""Preparation reuses raw inputs, never learned tensors or recurrent graphs."""
from copy import deepcopy
import random

import pytest
import torch

from brass_rl.config import PPOConfig
from brass_rl.returns import episode_targets
from brass_rl.sequences import sequence_batches
from test_ppo import assert_nested_equal, tiny as _tiny, toy_episodes


@pytest.fixture
def policy():
    return _tiny.__wrapped__()


def test_prepared_sequence_recomputes_learned_graph_and_exact_gradients(policy):
    import brass_rl.ppo as ppo
    model, components = policy
    episode = toy_episodes(model, components)[0]
    batch = next(sequence_batches([episode], episode_targets(episode), 64, 16, random.Random(1)))
    prepared = ppo._prepare_sequence(batch, components)
    for changed in (False, True):
        if changed:
            with torch.no_grad():
                model.event_gru.bias_ih.add_(1.)
        model.zero_grad(set_to_none=True)
        actual = ppo._sequence_stats(model, batch, components, prepared)
        (actual.values.sum()+actual.log_prob.sum()).backward()
        gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        fresh = ppo._sequence_stats(model, batch, components)
        assert torch.equal(actual.values, fresh.values)
        assert torch.equal(actual.log_prob, fresh.log_prob)
        (fresh.values.sum()+fresh.log_prob.sum()).backward()
        for gradient, p in zip(gradients, model.parameters()):
            assert (gradient is None and p.grad is None) or torch.equal(gradient, p.grad)
        assert model.event_gru.weight_ih.grad.abs().sum() > 0


def test_update_reuses_raw_encodings_with_exact_adam_metrics_and_shuffle(policy, monkeypatch):
    import brass_rl.ppo as ppo
    model, components = policy
    episodes = toy_episodes(model, components)
    initial = deepcopy(model.state_dict())
    config = PPOConfig(epochs=2)
    calls = {'public': 0, 'observation': 0}
    original_public, original_observation = ppo.encode_public, ppo.encode_observation
    def public(*args):
        calls['public'] += 1
        return original_public(*args)
    def observation(*args):
        calls['observation'] += 1
        return original_observation(*args)
    monkeypatch.setattr(ppo, 'encode_public', public)
    monkeypatch.setattr(ppo, 'encode_observation', observation)
    rng = random.Random(44)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics = ppo.update_policy(model, optimizer, episodes, config, rng)
    expected_model = deepcopy(model.state_dict())
    expected_optimizer, expected_rng = deepcopy(optimizer.state_dict()), rng.getstate()
    # Two complete sequences of two events and one decision, encoded only once.
    assert calls == {'public': 4, 'observation': 2}
    # The unprepared reference executes the identical original event/gradient order.
    monkeypatch.setattr(ppo, '_prepare_sequence', lambda *args: None)
    model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = random.Random(44)
    reference = ppo.update_policy(model, optimizer, episodes, config, rng)
    assert reference == metrics
    assert_nested_equal(model.state_dict(), expected_model)
    assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    assert rng.getstate() == expected_rng
    assert calls['public'] > 4 and calls['observation'] > 2


@pytest.mark.parametrize('backend', ['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='requires real CUDA'))])
def test_event_device_views_preserve_complete_inputs_and_reject_stale_spatial_keys(policy, backend):
    from dataclasses import fields, replace
    from brass_rl import ppo
    from brass_rl.device_events import place_events, staged_spatial_key
    from brass_rl.features import EncodedCandidates, EncodedPublic
    model, components = policy
    episode = toy_episodes(model, components)[0]
    batch = next(sequence_batches([episode], episode_targets(episode), 64, 16, random.Random(1)))
    raw = ppo._prepare_sequence(batch, components)
    original = [(batch, raw)]
    _, placed = place_events(original, torch.device(backend))[0]
    assert len(raw.main_events) == len(placed.main_events)
    assert placed.observations is raw.observations
    for before, after in zip(raw.main_events, placed.main_events):
        assert before.actor == after.actor and before.ply == after.ply and before.record is after.record
        for left, right, record in ((before.state, after.state, EncodedPublic),
                                    (before.delta, after.delta, EncodedPublic),
                                    (before.move, after.move, EncodedCandidates)):
            for f in fields(record):
                expected = torch.from_numpy(getattr(left, f.name))
                actual = getattr(right, f.name)
                assert torch.equal(actual.cpu(), expected) and actual.stride() == expected.stride()
                assert not actual.requires_grad
    board = placed.main_events[0].state
    assert staged_spatial_key(board) is not None
    assert staged_spatial_key(replace(board, slots=board.slots.clone())) is None
    from brass_rl.board_cache import BoardCache
    mixed = replace(board, locations=board.locations.cpu().numpy().copy())
    assert BoardCache(1024).key(model, mixed) is None
    board.slots.add_(.125)
    assert staged_spatial_key(board) is None
    untouched = [(batch, raw)] * 100
    for limit in (0, 1):
        assert place_events(untouched, torch.device(backend), limit=limit) is untouched
        assert all(pair is untouched[0] and pair[1] is raw for pair in untouched)


def test_optional_event_placement_oom_reverts_to_original_complete_batches(policy, monkeypatch):
    import brass_rl.device_events as device_events
    from brass_rl import ppo
    model, components = policy
    episode = toy_episodes(model, components)[0]
    batch = next(sequence_batches([episode], episode_targets(episode), 64, 16, random.Random(1)))
    original = [(batch, ppo._prepare_sequence(batch, components))]
    original *= 2
    first = original[0]
    stage = device_events._stage
    succeeded = []
    def fail_after_placement(events, device, remaining):
        if succeeded:
            raise torch.cuda.OutOfMemoryError()
        out = stage(events, device, remaining)
        if out[0] is not events:
            succeeded.append(True)
        return out
    monkeypatch.setattr(device_events, '_stage', fail_after_placement)
    assert device_events.place_events(original, torch.device('cpu')) is original
    assert succeeded == [True] and all(pair is first for pair in original)
