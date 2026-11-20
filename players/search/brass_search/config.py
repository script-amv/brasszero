"""Validated fixed-work budgets; importing configuration does not require Torch."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SearchConfig:
    root_candidates: int = 4
    worlds_per_round: int = 2
    rollout_depth: int = 8
    max_rollouts: int = 12
    value_scale: float = 10.0
    # Simulated actor/leaf scoring chunk. The actual root always uses native
    # candidate order and TorchAgent's fixed 1024 chunk for exact baseline draws.
    candidate_chunk: int = 1024
    wall_time_seconds: Optional[float] = None

    def __post_init__(self):
        for name in ('root_candidates', 'worlds_per_round', 'rollout_depth', 'max_rollouts', 'candidate_chunk'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'max_rollouts' else 1):
                raise ValueError(name + ' must be a valid integer count')
        for name in ('value_scale', 'wall_time_seconds'):
            value = getattr(self, name)
            if value is None and name == 'wall_time_seconds':
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
            object.__setattr__(self, name, float(value))

    def canonical_mapping(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate config key: ' + key)
                result[key] = value
            return result
        value = json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ValueError('search config must be an object')
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError('unknown search config field') from exc
