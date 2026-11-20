"""Compact shared graph/entity encoder, public-event memory and complete-move heads."""
from dataclasses import asdict
import hashlib
import json
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .board_cache import active_cache
from .graph_ops import graph_round_for
from .config import ModelConfig
from .distributions import PolicyOutput
from .features import EncodedCandidates, EncodedEvent, EncodedObservation, EncodedPublic, validate_events
from .schema import ComponentData, Schema


class BrassPolicy(nn.Module):
    def __init__(self, config: ModelConfig, schema: Schema, components: ComponentData):
        super().__init__()
        if schema.fingerprint != components.schema.fingerprint:
            raise ValueError("schema and components disagree")
        self.config, self.schema = config, schema
        e, c, h = config.entity_dim, config.context_dim, config.recurrent_dim
        dims = schema.dimensions
        identity = {"architecture": "complete-graph-gru-v1", "config": asdict(config),
                    "schema": schema.fingerprint}
        self.model_id = "brass-policy:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, count in (("location", dims["locations"]), ("slot", dims["slots"]),
                            ("link", dims["links"]), ("merchant", dims["merchants"]),
                            ("player", 4), ("industry", 6), ("card", dims["cards"])):
            setattr(self, name + "_ids", nn.Embedding(count, e))
        for name in ("location", "slot", "link", "merchant", "player"):
            setattr(self, name + "_features", nn.Linear(dims[name], e))
        self.global_features = nn.Linear(dims["global"], e)
        self.location_rounds = nn.ModuleList(nn.Linear(3 * e, e) for _ in range(config.message_passing_rounds))
        self.link_rounds = nn.ModuleList(nn.Linear(2 * e, e) for _ in range(config.message_passing_rounds))
        self.slot_rounds = nn.ModuleList(nn.Linear(2 * e, e) for _ in range(config.message_passing_rounds))
        self.merchant_rounds = nn.ModuleList(nn.Linear(2 * e, e) for _ in range(config.message_passing_rounds))
        # Flattened-board ablation remains compact: project shared 64-wide entities first.
        self.flat_scalar = nn.Linear(e, 1)
        flat_size = dims["locations"] + dims["slots"] + dims["links"] + dims["merchants"]
        self.flat_board = nn.Linear(flat_size, 4 * e)
        self.context = nn.Sequential(nn.Linear(10 * e + h, c), nn.Tanh())
        scalar_candidate = dims["candidate"] + dims["cards"] + 2 + 2 + dims["attributes"]
        self.candidate_scalars = nn.Linear(scalar_candidate, e)
        # scalar, industry, slot, location, link, merchant, owner, incidences,
        # three resource-source pools, and merchant beer pool.
        self.candidate_embedding = nn.Sequential(nn.Linear(12 * e, e), nn.Tanh())
        self.candidate_score = nn.Sequential(nn.Linear(c + e, e), nn.Tanh(), nn.Linear(e, 1))
        self.value_head = nn.Linear(c, 4)
        self.event_input = nn.Sequential(nn.Linear(20 * e, h), nn.Tanh())
        self.event_gru = nn.GRUCell(h, h)
        for name, array in (("slot_locations", components.slot_locations),
                             ("link_locations", components.link_locations),
                             ("link_location_mask", components.link_location_mask),
                             ("merchant_locations", components.merchant_locations)):
            self.register_buffer(name, torch.tensor(np.array(array, copy=True)))
        # Incidence matrices contain the engine's third farm endpoint too.
        for name, ids in (("slot_incidence", components.slot_locations[:, None]),
                           ("merchant_incidence", components.merchant_locations[:, None]),
                           ("link_incidence", components.link_locations)):
            matrix = torch.zeros((len(ids), dims["locations"]))
            for row, locations in enumerate(ids):
                for loc in locations:
                    if loc >= 0:
                        matrix[row, int(loc)] = 1.
            self.register_buffer(name, matrix)

    @property
    def device(self):
        return self.location_ids.weight.device

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(array, device=self.device)

    def initial_memory(self, batch: int, device) -> Tensor:
        if type(batch) is not int or batch < 1:
            raise ValueError("memory batch must be positive")
        return torch.zeros((batch, self.config.recurrent_dim), device=device)

    def _memory(self, memory: Tensor, batch: int) -> None:
        if (memory.shape != (batch, self.config.recurrent_dim) or memory.device != self.device
                or memory.dtype != torch.float32 or not torch.isfinite(memory).all().item()):
            raise ValueError("memory needs finite float32 [batch,recurrent_dim] on model device")

    def _entities(self, public: EncodedPublic) -> Tuple[Tensor, Dict[str, Tensor]]:
        cache = active_cache() if not torch.is_grad_enabled() else None
        key = cache.key(self, public) if cache is not None else None
        if key is not None:
            found = cache.get(key)
            if found is not None:
                spatial, board = found
                players = torch.tanh(self.player_features(self._tensor(public.players)) + self.player_ids.weight)
                context = torch.cat((torch.tanh(self.global_features(self._tensor(public.global_features))),
                                     players.flatten(), board))
                return context, dict(spatial, player=players)
        context, tables = self._encode_entities(public)
        if key is not None:
            cache.store(key, context, tables, self.config.entity_dim)
        return context, tables

    def _encode_entities(self, public: EncodedPublic) -> Tuple[Tensor, Dict[str, Tensor]]:
        entities = {}
        for name in ("location", "slot", "link", "merchant", "player"):
            layer = getattr(self, name + "_features")
            identities = getattr(self, name + "_ids").weight
            entities[name] = torch.tanh(layer(self._tensor(getattr(public, name + "s"))) + identities)
        loc, slots, links, merchants = (entities[n] for n in ("location", "slot", "link", "merchant"))
        if self.config.board_encoder == "graph":
            graph_round = graph_round_for(loc.device)
            for loc_layer, link_layer, slot_layer, merchant_layer in zip(
                    self.location_rounds, self.link_rounds, self.slot_rounds, self.merchant_rounds):
                loc, slots, links, merchants = graph_round(
                    loc, slots, links, merchants, self.slot_incidence, self.merchant_incidence,
                    self.link_incidence, self.slot_locations, self.merchant_locations,
                    loc_layer.weight, loc_layer.bias, link_layer.weight, link_layer.bias,
                    slot_layer.weight, slot_layer.bias, merchant_layer.weight, merchant_layer.bias)
            board = torch.cat([x.mean(0) for x in (loc, slots, links, merchants)])
        else:
            board = torch.tanh(self.flat_board(torch.cat([
                self.flat_scalar(x).squeeze(-1) for x in (loc, slots, links, merchants)])))
        entities.update(location=loc, slot=slots, link=links, merchant=merchants)
        public_context = torch.cat((torch.tanh(self.global_features(self._tensor(public.global_features))),
                                    entities["player"].flatten(), board))
        return public_context, entities

    def _gather(self, embedding: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
        return embedding[ids.clamp_min(0)] * mask.unsqueeze(-1)

    def _candidate(self, candidates: EncodedCandidates, start: int, end: int,
                   entities: Dict[str, Tensor]) -> Tensor:
        device = self.device
        data = {}
        def take(name):
            if name not in data:
                data[name] = torch.as_tensor(getattr(candidates, name)[start:end], device=device)
            return data[name]
        attributes = take("attribute_mask")
        scalar = self.candidate_scalars(torch.cat((take("features"), take("cards"),
            take("market_quantities"), take("market_mask"), attributes), dim=-1))
        pools = [scalar]
        for index, name in enumerate(("industry", "slot", "location", "link", "merchant")):
            table = self.industry_ids.weight if name == "industry" else entities[name]
            pools.append(self._gather(table, take(name + "_ids"), attributes[:, index]))
        pools.append(self._gather(self.player_ids.weight, take("target_owner_ids"), take("target_owner_mask")))
        incidence = self._gather(entities["location"], take("link_locations"), take("link_location_mask"))
        pools.append(incidence.sum(1) / take("link_location_mask").sum(1).clamp_min(1)[:, None])
        # Source quantities weight shared entity embeddings. Scatter to small owner/location
        # tables avoids materializing [candidates,slots,embedding] activation tensors.
        quantities = take("slot_sources") * take("source_mask").unsqueeze(-1)
        for channel in range(3):
            q = quantities[:, :, channel]
            owner_q = q.new_zeros((len(q), 4)).scatter_add(1, take("source_owners").clamp_min(0),
                                                          q * take("source_owner_mask"))
            loc_q = q.new_zeros((len(q), len(entities["location"]))).scatter_add(
                1, take("source_locations").clamp_min(0), q)
            pools.append(q @ entities["slot"] + owner_q @ self.player_ids.weight + loc_q @ entities["location"])
        pools.append((take("merchant_sources") * take("merchant_source_mask")) @ entities["merchant"])
        return self.candidate_embedding(torch.cat(pools, dim=-1))

    def update_memory(self, events: Sequence[EncodedEvent], memory: Tensor) -> Tensor:
        """Replay a chronological sequence for ONE viewer; never detach its graph.

        Call separately for each seat/sequence. Forward batches independent memories.
        Lifecycle/sequence builders validate absolute ply positions against boundaries.
        """
        self._memory(memory, 1)
        if not self.config.use_history:
            return torch.zeros_like(memory)
        validate_events(events)
        for event in events:
            state, entities = self._entities(event.state)
            delta, _ = self._entities(event.delta)
            move = self._candidate(event.move, 0, 1, entities).squeeze(0)
            actor = self.player_ids.weight[event.actor]
            inputs = self.event_input(torch.cat((state, delta, move, actor))).unsqueeze(0)
            memory = self.event_gru(inputs, memory)
        return memory

    def forward(self, observations: Sequence[EncodedObservation], memory: Tensor,
                candidate_chunk: int = 1024) -> PolicyOutput:
        if type(candidate_chunk) is not int or candidate_chunk < 1:
            raise ValueError("candidate_chunk must be a positive integer")
        if len(observations) == 0:
            raise ValueError("forward needs at least one observation")
        self._memory(memory, len(observations))
        logits, values, offsets = [], [], [0]
        for row, obs in enumerate(observations):
            if not isinstance(obs, EncodedObservation):
                raise TypeError("forward requires EncodedObservation")
            n = len(obs.candidates.features)
            if n == 0:
                raise ValueError("decision needs at least one candidate")
            public, entities = self._entities(obs)
            hand = self._tensor(obs.hand) @ self.card_ids.weight
            history = memory[row] if self.config.use_history else torch.zeros_like(memory[row])
            context = self.context(torch.cat((public, hand, history)))
            values.append(self.value_head(context).sigmoid())
            for start in range(0, n, candidate_chunk):
                end = min(n, start + candidate_chunk)
                # Bind slice/observation per closure: checkpoint recomputes after this loop.
                def score(ctx, loc, slots, links, merchants, players,
                          candidates=obs.candidates, first=start, last=end):
                    shared = dict(location=loc, slot=slots, link=links, merchant=merchants, player=players)
                    candidate = self._candidate(candidates, first, last, shared)
                    return self.candidate_score(torch.cat((ctx.expand(last - first, -1), candidate), dim=-1)).squeeze(-1)
                args = (context, *(entities[k] for k in ("location", "slot", "link", "merchant", "player")))
                if torch.is_grad_enabled() and n > candidate_chunk:
                    logits.append(checkpoint(score, *args, use_reentrant=False, preserve_rng_state=False))
                else:
                    logits.append(score(*args))
            offsets.append(offsets[-1] + n)
        flat = torch.cat(logits)
        return PolicyOutput(flat, torch.tensor(offsets, device=self.device),
                            torch.ones_like(flat, dtype=torch.bool), torch.stack(values))
