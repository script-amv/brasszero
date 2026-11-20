# SPDX-License-Identifier: Apache-2.0
"""Pin approved defaults and reject malformed training input before collection."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from brass_rl.config import EvalConfig, LeagueConfig, ModelConfig, PPOConfig, RuntimeConfig, TrainConfig

ROOT = Path(__file__).resolve().parents[2]


def test_default_hyperparameters():
    cfg = TrainConfig.from_json(ROOT / "training/configs/local-smoke.json")
    assert (cfg.ppo.gamma, cfg.ppo.gae_lambda, cfg.ppo.clip_epsilon) == (1., .95, .2)
    assert (cfg.ppo.learning_rate, cfg.ppo.entropy_coef, cfg.ppo.value_coef) == (.0003, .01, .5)
    assert (cfg.ppo.epochs, cfg.ppo.max_grad_norm, cfg.ppo.target_kl) == (4, .5, .03)
    assert (cfg.ppo.sequence_events, cfg.ppo.burn_in_events) == (64, 16)
    assert (cfg.concurrent_games, cfg.games_per_update, cfg.updates) == (2, 2, 2)
    assert (cfg.runtime.device, cfg.runtime.torch_threads, cfg.runtime.memory_budget_mib) == ("cpu", 1, 512)
    assert cfg.eval.interval_updates == 0
    assert cfg.master_seed is None
    assert cfg.league.lineup_probabilities == (.5, .35, .15)
    assert cfg.league.max_snapshots == 8
    assert (cfg.model.entity_dim, cfg.model.context_dim, cfg.model.recurrent_dim,
            cfg.model.message_passing_rounds) == (64, 128, 64, 2)


@pytest.mark.parametrize("section,field,value", [
    (None, "games_per_update", 0), (None, "concurrent_games", -1), (None, "updates", 0),
    (None, "games_per_update", True), (None, "concurrent_games", 2.5),
    (None, "master_seed", -1), (None, "master_seed", 2**64), (None, "master_seed", True),
    ("model", "entity_dim", 0), ("model", "context_dim", -1),
    ("model", "recurrent_dim", 1.5), ("model", "message_passing_rounds", 0),
    ("ppo", "gamma", .99), ("ppo", "gae_lambda", -0.1), ("ppo", "gae_lambda", 1.1),
    ("ppo", "clip_epsilon", 0), ("ppo", "clip_epsilon", 1),
    ("ppo", "learning_rate", 0), ("ppo", "learning_rate", float("inf")),
    ("ppo", "entropy_coef", -1), ("ppo", "entropy_coef", float("nan")),
    ("ppo", "epochs", 0), ("ppo", "max_grad_norm", 0), ("ppo", "target_kl", 0),
    ("ppo", "sequence_events", 0), ("ppo", "burn_in_events", -1),
    ("ppo", "value_coef", -1), ("ppo", "value_coef", "0.5"),
    ("league", "lineup_probabilities", [.5, .3, .1]),
    ("league", "lineup_probabilities", [1.1, -.1, 0]),
    ("league", "lineup_probabilities", [.5, .5]),
    ("league", "lineup_probabilities", [.5, float("nan"), .5]),
    ("league", "lineup_probabilities", "0.5,0.35,0.15"),
    ("league", "max_snapshots", 9), ("league", "max_snapshots", 0),
    ("eval", "interval_updates", -1),
    ("runtime", "torch_threads", 0), ("runtime", "memory_budget_mib", 0),
    ("runtime", "max_wall_time_seconds", -1), ("runtime", "budget_usd", -1),
    ("runtime", "max_wall_time_seconds", float("inf")),
    ("runtime", "budget_usd", float("nan")), ("runtime", "device", "tpu"),
])
def test_config_rejects_invalid_values(tmp_path, section, field, value):
    payload = {field: value} if section is None else {section: {field: value}}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        TrainConfig.from_json(path)


@pytest.mark.parametrize("section", [None, "model", "ppo", "league", "eval", "runtime"])
def test_unknown_fields_are_rejected(tmp_path, section):
    payload = {"typo": 1} if section is None else {section: {"typo": 1}}
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="typo"):
        TrainConfig.from_json(path)


@pytest.mark.parametrize("payload", [[], None, {"ppo": []}, {"runtime": None}])
def test_sections_must_be_objects(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="object"):
        TrainConfig.from_json(path)


def test_json_overrides_and_immutable_validated_constructors(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"master_seed": 2**64 - 1, "games_per_update": 32,
                               "concurrent_games": 8, "runtime": {"budget_usd": 0},
                               "league": {"lineup_probabilities": [1, 0, 0]}}))
    cfg = TrainConfig.from_json(path)
    assert (cfg.games_per_update, cfg.concurrent_games, cfg.master_seed) == (32, 8, 2**64 - 1)
    assert cfg.league.lineup_probabilities == (1, 0, 0)
    assert cfg.runtime.budget_usd == 0
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.ppo, PPOConfig)
    assert isinstance(cfg.league, LeagueConfig)
    assert isinstance(cfg.eval, EvalConfig)
    assert isinstance(cfg.runtime, RuntimeConfig)
    with pytest.raises(FrozenInstanceError):
        cfg.games_per_update = 0
    with pytest.raises(ValueError, match="gamma"):
        PPOConfig(gamma=.9)
    with pytest.raises(ValueError, match="ppo"):
        TrainConfig(ppo={})


@pytest.mark.parametrize("value", [0, 16])
def test_burn_in_boundaries_are_accepted(tmp_path, value):
    assert PPOConfig(burn_in_events=value).burn_in_events == value
    path = tmp_path / "burn-in.json"
    path.write_text(json.dumps({"ppo": {"burn_in_events": value}}))
    assert TrainConfig.from_json(path).ppo.burn_in_events == value


def test_burn_in_above_limit_is_rejected_by_constructor():
    with pytest.raises(ValueError, match="burn_in_events"):
        PPOConfig(burn_in_events=17)


def test_burn_in_above_limit_is_rejected_by_json_loader(tmp_path):
    path = tmp_path / "burn-in.json"
    path.write_text(json.dumps({"ppo": {"burn_in_events": 17}}))
    with pytest.raises(ValueError, match="burn_in_events"):
        TrainConfig.from_json(path)
