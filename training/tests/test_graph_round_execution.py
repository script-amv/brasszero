# SPDX-License-Identifier: Apache-2.0
"""Exercise the spatial block after warm inference and during recurrent training."""
import brass
import torch

from brass_rl.config import ModelConfig
from brass_rl.features import encode_observation
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData


def test_warmed_cpu_graph_round_supports_repeated_differentiable_boards():
    components = ComponentData.load()
    model = BrassPolicy(ModelConfig(), components.schema, components)
    game = brass.Game(5)
    board = encode_observation(game.observation(game.current_player()), components)
    with torch.no_grad():
        for _ in range(6):
            model._encode_entities(board)
    for _ in range(12):
        context, _ = model._encode_entities(board)
        context.sum().backward()
        assert torch.isfinite(model.location_rounds[0].weight.grad).all()
        model.zero_grad()
