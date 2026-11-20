"""One seat's private public-history memory and policy-sampling stream."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import brass
import torch
from torch import Tensor

from brass_ai.events import PublicEvent
from .distributions import PolicyOutput, policy_probabilities, policy_stats
from .features import EncodedObservation, _encode_transition, _public_json, encode_observation
from .model import BrassPolicy
from .schema import ComponentData


@dataclass(frozen=True)
class PreparedDecision:
    encoded: EncodedObservation
    memory: Tensor
    ply: int


@dataclass(frozen=True)
class Choice:
    index: int
    log_prob: float
    values: Tuple[float, float, float, float]


def _actor(public):
    phase = public["phase"]
    if public["game_over"]:
        return None
    return phase["Shortfall"]["player"] if isinstance(phase, dict) else public["turn_order"][public["turn_idx"]]


class TorchAgent:
    def __init__(self, model: BrassPolicy, components: ComponentData, seed: int, mode: str = "sample"):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("policy sampling seed must be an unsigned 64-bit integer")
        if mode not in ("sample", "argmax"):
            raise ValueError("mode must be sample or argmax")
        if model.schema.fingerprint != components.schema.fingerprint:
            raise ValueError("agent model/component schema mismatch")
        self.model, self.components, self.seed, self.mode = model, components, seed, mode
        self.name = "torch:" + model.model_id + ":" + mode
        self._seat = None  # type: Optional[int]
        self._memory = None  # type: Optional[Tensor]
        self._prepared = None
        self._awaiting = None
        self._generator = torch.Generator(device="cpu")
        self.ply = 0
        self.latest_public_json = None

    def reset(self, seat: int, initial_public_json: str) -> None:
        if type(seat) is not int or seat not in range(4):
            raise ValueError("seat must be in 0..3")
        public = _public_json(initial_public_json, self.components)
        # Sampling seeds never contain a deal seed. Each seat also has a distinct stream.
        digest = hashlib.blake2b(self.seed.to_bytes(8, "little") + bytes([seat]),
                                 person=b"brass-policy-rng", digest_size=8).digest()
        self._generator.manual_seed(int.from_bytes(digest, "little"))
        self._seat, self._public = seat, public
        self._encoded_public = None
        self.latest_public_json = initial_public_json
        self.ply = 0
        self._memory = self.model.initial_memory(1, self.model.device)
        self._prepared, self._awaiting = None, None

    def _reset_required(self) -> None:
        if self._seat is None:
            raise RuntimeError("reset is required before observations or decisions")

    def observe_public(self, event: PublicEvent) -> None:
        self._reset_required()
        if not isinstance(event, PublicEvent):
            raise TypeError("observe_public requires PublicEvent")
        if type(event.ply) is not int or event.ply != self.ply + 1:
            raise ValueError("public event ply is duplicate, missing, or out of order")
        if self._prepared is not None:
            raise RuntimeError("prepared decision must be chosen before its public event")
        if self._awaiting is not None:
            if event.actor != self._seat or json.loads(event.move_json) != json.loads(self._awaiting):
                raise ValueError("public event does not match chosen move and seat")
        if event.actor != _actor(self._public):
            raise ValueError("public event actor mismatches previous public state")
        encoded, public = _encode_transition(event, self._seat, None, self.components, self._public, self._encoded_public)
        with torch.no_grad():
            updated = self.model.update_memory([encoded], self._memory)
        # Commit lifecycle state only after every validation and model update succeeds.
        self._memory = updated.detach()
        self._public, self._encoded_public = public, encoded.state
        self.latest_public_json = event.public_json
        self.ply = event.ply
        self._awaiting = None

    def prepare(self, obs: brass.Observation) -> PreparedDecision:
        self._reset_required()
        if self._awaiting is not None:
            raise RuntimeError("chosen decision is awaiting its public event")
        if self._prepared is not None:
            raise RuntimeError("a decision is already prepared")
        if not isinstance(obs, brass.Observation):
            raise TypeError("prepare requires native brass.Observation")
        if obs.me != self._seat:
            raise ValueError("observation viewer does not match agent seat")
        if _public_json(obs.public_json(), self.components) != self._public:
            raise ValueError("observation public snapshot mismatches latest consumed history")
        if _actor(self._public) != self._seat:
            raise ValueError("observation is not this seat's decision")
        if obs.num_legal() == 0:
            raise ValueError("observation has no legal candidates")
        encoded = encode_observation(obs, self.components)
        prepared = PreparedDecision(encoded, self.memory_snapshot(), self.ply)
        self._prepared = (prepared, tuple(obs.legal_moves()))
        return prepared

    def choose(self, output: PolicyOutput, row: int) -> Choice:
        self._reset_required()
        if self._awaiting is not None:
            raise RuntimeError("chosen decision is awaiting its public event")
        if self._prepared is None:
            raise RuntimeError("prepare is required before choose")
        if type(row) is not int or not 0 <= row < len(output.values):
            raise ValueError("output row does not exist")
        probabilities = policy_probabilities(output)
        start, end = output.offsets[row:row + 2].detach().cpu().tolist()
        moves = self._prepared[1]
        if end - start != len(moves) or not output.valid[start:end].all().item():
            raise ValueError("output row must contain every prepared legal candidate")
        p = probabilities[start:end].detach().cpu()
        index = int(p.argmax().item() if self.mode == "argmax" else
                    torch.multinomial(p, 1, generator=self._generator).item())
        # Slice one decision to avoid inventing choices for unrelated batch rows.
        local = PolicyOutput(output.logits[start:end], torch.tensor([0, end - start], device=output.offsets.device),
                             output.valid[start:end], output.values[row:row + 1])
        stats = policy_stats(local, torch.tensor([index], dtype=torch.int64))
        value_tuple = tuple(float(v) for v in stats.values[0].detach().cpu().tolist())
        if not torch.isfinite(stats.values).all().item():
            raise ValueError("policy values must be finite")
        choice = Choice(index, float(stats.log_prob[0].detach().cpu().item()), value_tuple)
        self._awaiting = moves[index]
        self._prepared = None
        return choice

    def memory_snapshot(self) -> Tensor:
        self._reset_required()
        return self._memory.detach().clone()

    def act(self, obs: brass.Observation) -> int:
        with torch.no_grad():
            prepared = self.prepare(obs)
            output = self.model([prepared.encoded], prepared.memory, candidate_chunk=1024)
            return self.choose(output, 0).index


def load_agent(path: Path, seed: int, device: str = "cpu", mode: str = "sample") -> TorchAgent:
    """Load a validated policy without ever deserializing training controller data."""
    from .policy_io import load_policy
    model, components, metadata = load_policy(path, device)
    agent = TorchAgent(model, components, seed, mode)
    agent.replay_identity = metadata.id
    return agent


def register_checkpoint(name: str, path: Path, device: str = "cpu", mode: str = "sample") -> None:
    """Register frozen shared weights; every factory call creates private seat state."""
    from brass_ai.agents import register
    from .policy_io import load_policy
    if mode not in ("sample", "argmax"):
        raise ValueError("mode must be sample or argmax")
    model, components, metadata = load_policy(path, device)
    def factory(seed):
        agent = TorchAgent(model, components, seed, mode)
        agent.replay_identity = metadata.id
        return agent
    register(name, factory)
