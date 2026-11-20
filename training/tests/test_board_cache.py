# SPDX-License-Identifier: Apache-2.0
"""Identical spatial tensors can be shared only inside the same fixed-weight scope."""
from dataclasses import replace

import numpy as np
import torch

from brass_rl.board_cache import active_cache, fixed_boards
from brass_rl.config import ModelConfig
from brass_rl.features import encode_observation
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData
from helpers import native_game


def test_cached_spatial_slice_keeps_current_player_and_global_inputs_and_versions(monkeypatch):
    torch.set_num_threads(1)
    components = ComponentData.load()
    model = BrassPolicy(ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4),
                        components.schema, components)
    obs = encode_observation(native_game().observation(0), components)
    changed = replace(obs, players=obs.players+np.float32(.25), global_features=obs.global_features+np.float32(.125))
    computations = []
    original = model._encode_entities
    def counted(public):
        computations.append(1)
        return original(public)
    monkeypatch.setattr(model, "_encode_entities", counted)
    with torch.no_grad():
        expected, _ = model._entities(changed)
        with fixed_boards() as cache:
            model._entities(obs)
            actual, _ = model._entities(changed)
            assert torch.equal(actual, expected) and len(computations) == 2 and cache.used > 0
            model.location_ids.weight.add_(.125)
            new, _ = model._entities(changed)
            assert len(computations) == 3 and not torch.equal(actual, new)
            cache.limit_to(0)
            assert cache.used == 0 and not cache.entries
            exact, _ = model._entities(changed)
            assert torch.equal(new, exact) and len(computations) == 4
        assert active_cache() is None


def test_differentiable_scalar_board_work_never_reuses_a_cached_graph():
    components = ComponentData.load()
    model = BrassPolicy(ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4),components.schema,components)
    obs = encode_observation(native_game().observation(0), components)
    with fixed_boards():
        with torch.no_grad():
            model._entities(obs)
        public, _ = model._entities(obs)
        public.sum().backward()
        assert model.location_ids.weight.grad is not None
        assert model.location_ids.weight.grad.abs().sum() > 0


def test_only_cpu_selects_the_numerically_verified_compiled_graph():
    from brass_rl.graph_ops import eager_graph_round, graph_round_for
    with torch.no_grad():
        assert isinstance(graph_round_for(torch.device("cpu")), torch.jit.ScriptFunction)
    for backend in ("cuda", "mps"):
        assert graph_round_for(torch.device(backend)) is eager_graph_round
