# SPDX-License-Identifier: Apache-2.0
"""Model contracts catch candidate loss, wrong topology and severed gradients."""
from dataclasses import fields, replace
import json

import numpy as np
import pytest
import torch

from brass_ai.events import PublicEvent
from brass_rl.config import ModelConfig
from brass_rl.distributions import policy_probabilities, policy_stats
from brass_rl.features import encode_candidates, encode_observation, encode_public
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData
from helpers import native_game, case_move


@pytest.fixture(scope="module")
def components():
    torch.set_num_threads(1)
    return ComponentData.load()


def policy(components, **kwargs):
    torch.manual_seed(17)
    return BrassPolicy(ModelConfig(**kwargs), components.schema, components)


def test_policy_legal_normalized(components):
    model = policy(components)
    obs = encode_observation(native_game().observation(0), components)
    out = model([obs, obs], model.initial_memory(2, "cpu"), candidate_chunk=31)
    assert out.logits.shape == (2 * len(obs.candidates.features),)
    assert out.values.shape == (2, 4)
    assert ((out.values >= 0) & (out.values <= 1)).all()
    p = policy_probabilities(out)
    assert p[:out.offsets[1]].sum().item() == pytest.approx(1.)
    assert torch.isfinite(policy_stats(out, torch.tensor([0, 0])).entropy).all()
    assert sum(p.numel() for p in model.parameters()) < 1_000_000


def test_chunked_matches_full_logits_statistics_and_gradients(components):
    model = policy(components)
    obs = encode_observation(native_game().observation(0), components)
    indices = np.arange(8193) % len(obs.candidates.features)
    candidates = replace(obs.candidates, **{f.name: getattr(obs.candidates, f.name)[indices].copy()
                                           for f in fields(obs.candidates)})
    candidates.market_quantities[:, 0] += np.arange(8193, dtype=np.float32) / 8193
    obs = replace(obs, candidates=candidates)
    def run(chunk):
        model.zero_grad(set_to_none=True)
        out = model([obs], model.initial_memory(1, "cpu"), candidate_chunk=chunk)
        stats = policy_stats(out, torch.tensor([8192]))
        (-stats.log_prob.sum() - .01 * stats.entropy.sum() + stats.values.sum()).backward()
        return out.logits.detach(), stats.log_prob.detach(), stats.entropy.detach(), {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    full, chunked = run(8193), run(127)
    for a, b in zip(full[:3], chunked[:3]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert full[3].keys() == chunked[3].keys()
    for key in full[3]:
        torch.testing.assert_close(full[3][key], chunked[3][key], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(chunked[1]).all()
    assert chunked[3]["candidate_score.2.weight"].abs().sum() > 0


def test_candidate_order_equivariance(components):
    model = policy(components)
    obs = encode_observation(native_game().observation(0), components)
    perm = np.random.default_rng(71).permutation(len(obs.candidates.features))
    shuffled = replace(obs, candidates=replace(obs.candidates, **{
        f.name: getattr(obs.candidates, f.name)[perm] for f in fields(obs.candidates)}))
    a = model([obs], model.initial_memory(1, "cpu"), 31)
    b = model([shuffled], model.initial_memory(1, "cpu"), 31)
    torch.testing.assert_close(a.logits[perm], b.logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a.values, b.values)


def test_third_location_link_embedded(components):
    model = policy(components)
    obs = encode_observation(native_game().observation(0), components)
    public = json.loads(native_game().observation(0).public_json())
    obs = replace(obs, candidates=encode_candidates([case_move("Canal")], 0, public, components))
    assert components.link_locations[29, 2] >= 0
    farm = int(components.link_locations[29, 2])
    # This farm touches only link29. Inspect its link embedding so global pooling
    # cannot make a missing third incidence appear to work.
    _, entities = model._entities(obs)
    entities["link"][29].sum().backward()
    assert model.location_ids.weight.grad[farm].abs().sum() > 0
    out = model([obs], model.initial_memory(1, "cpu"), 1)
    with torch.no_grad():
        model.location_ids.weight[farm].add_(1.)
    changed = model([obs], model.initial_memory(1, "cpu"), 1)
    assert not torch.allclose(out.logits, changed.logits)


def test_sources_and_quantities_affect_score(components):
    model = policy(components)
    game = native_game()
    public = json.loads(game.observation(0).public_json())
    moves = [case_move("Build") for _ in range(3)]
    moves[1]["Build"]["coal"]["from_tiles"] = [[5, 2]]
    moves[2]["Build"]["coal"]["from_tiles"] = [[1, 3]]
    obs = replace(encode_observation(game.observation(0), components),
                  candidates=encode_candidates(moves, 0, public, components))
    logits = model([obs], model.initial_memory(1, "cpu"), 2).logits
    assert not torch.isclose(logits[0], logits[1], atol=1e-7)
    assert not torch.isclose(logits[0], logits[2], atol=1e-7)


def test_memory_replay_gradients_and_labeled_ablations(components):
    game = native_game()
    before = game.observation(0).public_json()
    move = game.observation(0).legal_moves()[0]
    game.apply_index(0)
    event = encode_public(PublicEvent(1, 0, move, game.observation(0).public_json()), 0, before, components)
    model = policy(components)
    h = model.update_memory([event, replace(event, ply=2)], model.initial_memory(1, "cpu"))
    h2 = model.update_memory([replace(event, ply=2)], model.update_memory([event], model.initial_memory(1, "cpu")))
    torch.testing.assert_close(h, h2)
    assert not torch.equal(h, model.initial_memory(1, "cpu"))
    h.sum().backward()
    assert model.event_gru.weight_hh.grad.abs().sum() > 0
    assert model.slot_features.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="single|batch"):
        model.update_memory([event], model.initial_memory(2, "cpu"))
    no_history = policy(components, use_history=False)
    flat = policy(components, board_encoder="flat")
    assert len({model.model_id, no_history.model_id, flat.model_id}) == 3
    assert torch.equal(no_history.update_memory([event], no_history.initial_memory(1, "cpu")),
                       no_history.initial_memory(1, "cpu"))
    obs = encode_observation(game.observation(game.current_player()), components)
    assert torch.isfinite(flat([obs], flat.initial_memory(1, "cpu"), 31).logits).all()


@pytest.mark.parametrize("kwargs", [{"board_encoder": "other"}, {"use_history": 1}])
def test_invalid_ablations_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_bad_forward_inputs_fail_without_truncating(components):
    model = policy(components)
    obs = encode_observation(native_game().observation(0), components)
    for chunk in (0, -1, True):
        with pytest.raises(ValueError, match="chunk"):
            model([obs], model.initial_memory(1, "cpu"), chunk)
    with pytest.raises(ValueError, match="memory"):
        model([obs], model.initial_memory(2, "cpu"), 2)
    terminal = encode_observation(native_game("GameOver").observation(0), components)
    with pytest.raises(ValueError, match="candidate"):
        model([terminal], model.initial_memory(1, "cpu"), 2)


def test_packed_events_reject_nonintegral_actor_and_extra_move_rows(components):
    game = native_game()
    before = game.observation(0).public_json()
    choice = game.observation(0).legal_moves()[0]
    game.apply_index(0)
    event = encode_public(PublicEvent(1, 0, choice, game.observation(0).public_json()), 0, before, components)
    model = policy(components)
    memory = model.initial_memory(1, "cpu")
    with pytest.raises(ValueError, match="actor"):
        model.update_memory([replace(event, actor=1.5)], memory)
    extra = replace(event.move, **{f.name: np.repeat(getattr(event.move, f.name), 2, axis=0)
                                  for f in fields(event.move)})
    with pytest.raises(ValueError, match="one candidate"):
        model.update_memory([replace(event, move=extra), replace(event, ply=2)], memory)
