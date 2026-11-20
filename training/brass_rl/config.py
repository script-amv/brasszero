"""Immutable Python 3.9 configuration, usable without optional numerical packages."""
from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Optional, Tuple


def _integer(name, value, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("%s must be an integer in [%s, %s]" % (name, minimum, maximum or "unbounded"))


def _number(name, value, minimum=0, maximum=None, exclusive_min=False, exclusive_max=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum or (exclusive_min and value == minimum)
            or (maximum is not None and (value > maximum or (exclusive_max and value == maximum)))):
        raise ValueError("%s is outside its finite allowed range" % name)


def _device(value):
    if value not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")


@dataclass(frozen=True)
class ModelConfig:
    entity_dim: int = 64
    context_dim: int = 128
    recurrent_dim: int = 64
    message_passing_rounds: int = 2
    board_encoder: str = "graph"
    use_history: bool = True

    def __post_init__(self):
        for name in ("entity_dim", "context_dim", "recurrent_dim", "message_passing_rounds"):
            _integer("model." + name, getattr(self, name))
        if self.board_encoder not in ("graph", "flat"):
            raise ValueError("model.board_encoder must be graph or flat")
        if type(self.use_history) is not bool:
            raise ValueError("model.use_history must be boolean")


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.
    gae_lambda: float = .95
    clip_epsilon: float = .2
    learning_rate: float = .0003
    entropy_coef: float = .01
    epochs: int = 4
    max_grad_norm: float = .5
    target_kl: float = .03
    sequence_events: int = 64
    burn_in_events: int = 16
    value_coef: float = .5
    reward: str = "strict_win"
    critic_target: str = "gae"

    def __post_init__(self):
        if self.reward not in ("strict_win", "score") or self.critic_target not in ("gae", "terminal"):
            raise ValueError("unknown PPO reward or critic target")
        _number("ppo.gamma", self.gamma, 1, 1)
        _number("ppo.gae_lambda", self.gae_lambda, 0, 1)
        _number("ppo.clip_epsilon", self.clip_epsilon, 0, 1, True, True)
        for name in ("learning_rate", "max_grad_norm", "target_kl"):
            _number("ppo." + name, getattr(self, name), exclusive_min=True)
        for name in ("entropy_coef", "value_coef"):
            _number("ppo." + name, getattr(self, name))
        for name in ("epochs", "sequence_events"):
            _integer("ppo." + name, getattr(self, name))
        _integer("ppo.burn_in_events", self.burn_in_events, 0, 16)


@dataclass(frozen=True)
class LeagueConfig:
    # Four current copies; current vs three frozen; current vs mixed opponents.
    lineup_probabilities: Tuple[float, float, float] = (.5, .35, .15)
    max_snapshots: int = 8
    snapshot_interval_updates: int = 10
    holdout_update: int = 5
    lineup_schedule: str = "sample"

    def __post_init__(self):
        probabilities = self.lineup_probabilities
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 3:
            raise ValueError("league.lineup_probabilities must contain three probabilities")
        for value in probabilities:
            _number("league.lineup_probabilities", value, 0, 1)
        if not math.isclose(sum(probabilities), 1., rel_tol=0., abs_tol=1e-9):
            raise ValueError("league.lineup_probabilities must sum to one")
        object.__setattr__(self, "lineup_probabilities", tuple(probabilities))
        if self.lineup_schedule not in ("sample", "four_of_five"):
            raise ValueError("unknown league lineup schedule")
        if self.lineup_schedule == "four_of_five" and tuple(probabilities) != (.8, .2, 0):
            raise ValueError("four_of_five requires the declared 80/20 lineup")
        _integer("league.max_snapshots", self.max_snapshots, 1, 8)
        _integer("league.snapshot_interval_updates", self.snapshot_interval_updates)
        _integer("league.holdout_update", self.holdout_update)


@dataclass(frozen=True)
class EvalConfig:
    # Zero disables periodic strength evaluation in the local smoke profile.
    interval_updates: int = 0
    deals: int = 2
    replay_limit: int = 10
    training_replay_limit: int = 10

    def __post_init__(self):
        _integer("eval.interval_updates", self.interval_updates, 0)
        _integer("eval.deals", self.deals, 1, 100000)
        _integer("eval.replay_limit", self.replay_limit, 0, 100)
        _integer("eval.training_replay_limit", self.training_replay_limit, 0, 100)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    torch_threads: int = 1
    memory_budget_mib: int = 512
    max_wall_time_seconds: Optional[float] = None
    checkpoint_margin_seconds: float = 30.
    budget_usd: Optional[float] = None

    def __post_init__(self):
        _device(self.device)
        _integer("runtime.torch_threads", self.torch_threads)
        _integer("runtime.memory_budget_mib", self.memory_budget_mib)
        _number("runtime.checkpoint_margin_seconds", self.checkpoint_margin_seconds)
        for name in ("max_wall_time_seconds", "budget_usd"):
            if getattr(self, name) is not None:
                _number("runtime." + name, getattr(self, name))


_SECTIONS = {"model": ModelConfig, "ppo": PPOConfig, "league": LeagueConfig,
             "eval": EvalConfig, "runtime": RuntimeConfig}


def _from_mapping(cls, payload, name):
    if not isinstance(payload, dict):
        raise ValueError("%s must be a JSON object" % name)
    unknown = payload.keys() - {entry.name for entry in fields(cls)}
    if unknown:
        raise ValueError("%s has unknown fields: %s" % (name, ", ".join(sorted(unknown))))
    return cls(**payload)


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    games_per_update: int = 2
    concurrent_games: int = 2
    updates: int = 2
    master_seed: Optional[int] = None

    def __post_init__(self):
        for name, cls in _SECTIONS.items():
            if not isinstance(getattr(self, name), cls):
                raise ValueError("%s must be a %s" % (name, cls.__name__))
        for name in ("games_per_update", "concurrent_games", "updates"):
            _integer(name, getattr(self, name))
        if self.master_seed is not None:
            _integer("master_seed", self.master_seed, 0, 2**64 - 1)

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("train config must be a JSON object")
        converted = dict(payload)
        for name, section in _SECTIONS.items():
            if name in converted:
                converted[name] = _from_mapping(section, converted[name], name)
        return _from_mapping(cls, converted, "train config")
