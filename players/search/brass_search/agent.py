"""Frozen root-search player, with TorchAgent's exact live sampling lifecycle."""
import copy
import math
from pathlib import Path
import time

import torch

from brass_rl.agent import TorchAgent
from .allocation import allocate, root_candidates
from .config import SearchConfig
from .identity import search_identity, player_manifest
from .inference import advance_memories, domain_seed, forward, move_key, standard_gumbel
from .simulator import SampledSimulator


class SearchAgent(TorchAgent):
    def __init__(self, model, components, seed, config, *, base_policy_id):
        if not isinstance(config, SearchConfig):
            raise ValueError('config must be a validated SearchConfig')
        super().__init__(model, components, seed, mode='sample')
        self.config, self.base_policy_id = config, base_policy_id
        self.replay_identity = search_identity(base_policy_id, components.schema.fingerprint, config)
        self.name = self.replay_identity
        self.last_diagnostics = {}
        self._memories = None

    def reset(self, seat, initial_public_json):
        super().reset(seat, initial_public_json)
        self._memories = [self.model.initial_memory(1, self.model.device).detach() for _ in range(4)]
        self.last_diagnostics = {}

    def public_memory_snapshots(self):
        self._reset_required()
        return tuple(memory.detach().clone() for memory in self._memories)

    def observe_public(self, event):
        self._reset_required()
        # Run inherited lifecycle validation/update on staged attributes. This
        # preserves its exact root computation without exposing a partial commit.
        staged = copy.copy(self)
        TorchAgent.observe_public(staged, event)
        if not torch.isfinite(staged._memory).all().item():
            raise ValueError('public memory must be finite')
        updated = advance_memories(self.model, self.components, self._memories,
                                   event, self.latest_public_json, skip_viewer=self._seat)
        updated[self._seat] = staged._memory
        self._memory, self._public = staged._memory, staged._public
        self.latest_public_json, self.ply = staged.latest_public_json, staged.ply
        self._awaiting = staged._awaiting
        self._memories = updated

    def act(self, obs):
        # A decision is a transaction: an error must leave neither a consumed
        # baseline draw nor an expected move that was never returned to the referee.
        staged = copy.copy(self)
        staged._generator = torch.Generator(device='cpu')
        staged._generator.set_state(self._generator.get_state())
        selected = staged._act(obs)
        self._generator.set_state(staged._generator.get_state())
        self._prepared, self._awaiting = staged._prepared, staged._awaiting
        self.last_diagnostics = staged.last_diagnostics
        return selected

    def _act(self, obs):
        start = time.monotonic()
        deadline = None if self.config.wall_time_seconds is None else start + self.config.wall_time_seconds
        counters = dict(native_applies=0, inference_calls=0, policy_forward_calls=0,
                        memory_update_calls=0, root_candidates_encoded=0, rollout_candidates_encoded=0,
                        max_live_worlds=0, max_live_continuations=0, worlds_sampled=0)
        with torch.no_grad():
            prepared = self.prepare(obs)
            # The one root forward preserves TorchAgent batch/order exactly. Search
            # canonicalizes these completed scores, never the baseline model input.
            output = forward(self.model, self.components, obs, prepared.memory, self.config, counters,
                             encoded=prepared.encoded, root=True)
            fallback_index = self.choose(output, 0).index  # Exactly ONE live baseline draw.
            moves = obs.legal_moves()
            keys = [move_key(move) for move in moves]
            if len(keys) != len(set(keys)):
                raise ValueError('duplicate root legal move')
            fallback = keys[fallback_index]
            order = sorted(range(len(keys)), key=lambda i: keys[i])
            canonical_logits = output.logits[order].detach().cpu().double()
            logp = torch.log_softmax(canonical_logits, dim=0).tolist()
            gseed = domain_seed(self.seed, self._seat, self.ply, 'gumbel')
            canonical_keys = [keys[i] for i in order]
            candidates = root_candidates(canonical_keys, logp,
                                         [standard_gumbel(gseed, k) for k in canonical_keys],
                                         min(self.config.root_candidates, len(keys)))
            simulator = SampledSimulator(self.model, self.components, obs, self.public_memory_snapshots(),
                                         self.seed, self.ply, self.config, counters, deadline)
            result = allocate(candidates, self.config, simulator, fallback)
            selected = keys.index(result.selected)
            # choose() awaited the baseline sample; only the actual move is expected.
            self._awaiting = moves[selected]
        elapsed = time.monotonic() - start
        candidate_keys = {c.key for c in candidates}
        mass = math.fsum(math.exp(p) for k, p in zip(canonical_keys, logp) if k in candidate_keys)
        reason = 'single_legal' if len(keys) == 1 and self.config.max_rollouts else result.reason
        self.last_diagnostics = dict(counters, rollouts=result.rollouts,
            completed_rounds=result.completed_rounds, discarded_rollouts=result.discarded_rollouts,
            reason=reason, fallback=result.reason not in ('completed', 'single_candidate'),
            legal_count=len(keys), selected_root_count=len(candidates), selected_root_prior_mass=mass,
            elapsed_seconds=elapsed, wall_overrun_seconds=0. if deadline is None else max(0., time.monotonic()-deadline))
        return selected


def load_search_agent(path, seed, config, device='cpu'):
    """Reject malformed config before loading an unchanged validated portable policy."""
    if not isinstance(config, SearchConfig):
        raise ValueError('config must be a validated SearchConfig')
    from brass_rl.policy_io import load_policy
    model, components, metadata = load_policy(Path(path), device)
    agent = SearchAgent(model, components, seed, config, base_policy_id=metadata.id)
    agent.manifest = player_manifest(metadata, config)
    agent.base_update = metadata.update
    return agent


def register_search_checkpoint(name, path, config, device='cpu'):
    """Share frozen weights only; each registered seat has private memory/RNG."""
    if not isinstance(config, SearchConfig):
        raise ValueError('config must be a validated SearchConfig')
    from brass_ai.agents import register
    from brass_rl.policy_io import load_policy
    model, components, metadata = load_policy(Path(path), device)
    manifest = player_manifest(metadata, config)
    def factory(seed):
        agent = SearchAgent(model, components, seed, config, base_policy_id=metadata.id)
        agent.manifest = dict(manifest)
        agent.base_update = metadata.update
        return agent
    register(name, factory)
