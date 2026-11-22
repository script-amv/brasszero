# SPDX-License-Identifier: Apache-2.0
"""Synchronous frozen-policy collection of complete, reproducible four-seat games.

The controller alone owns referees/seeds/lineups. Neural seats receive native
observations and sanitized public events through TorchAgent's lifecycle only.
Memory limits are conservative allocation estimates, not process RSS limits.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import brass
import numpy as np
import torch
from torch import Tensor

from brass_ai.arena import GameResult, RefereeGame
from brass_ai.events import PublicEvent
from brass_ai.replay_archive import Recorder, labeled_seats, recording_metadata
from .board_cache import active_cache, fixed_boards
from .agent import PreparedDecision, TorchAgent
from .config import TrainConfig
from .league import replay_identity, League, Lineup, Snapshot
from .schema import ComponentData
from .seeding import SeedStreams
from .replay import experiment_identity


@dataclass(frozen=True)
class Decision:
    game_id: int
    ply: int
    seat: int
    observation: brass.Observation
    index: int
    old_log_prob: float
    old_values: Tuple[float, float, float, float]

    @property
    def key(self) -> Tuple[int, int, int]:
        """Stable target/reconstruction key: (game ID, pre-event ply, seat)."""
        return self.game_id, self.ply, self.seat


@dataclass(frozen=True)
class Episode:
    game_id: int
    initial_public_json: str
    events: Tuple[PublicEvent, ...]
    boundary_states: Dict[int, Tuple[Tensor, ...]]
    decisions: Tuple[Decision, ...]
    lineup: Lineup
    result: GameResult
    replay_id: Optional[str] = None


class CollectionMemoryError(RuntimeError):
    """A complete allocation cannot fit; collection has not silently lost games."""


class _MemoryBudget:
    def __init__(self, limit: int):
        self.limit, self.used = limit, 0
        self.cache = active_cache()

    def fits(self, size: int) -> bool:
        remaining = self.limit - self.used - size
        if self.cache is not None:
            self.cache.limit_to(remaining)
        return remaining >= 0

    def check(self, size: int, context: str) -> None:
        if not self.fits(size):
            raise CollectionMemoryError(
                "collection memory budget exceeded allocating %s: retained=%d, requested=%d, budget=%d bytes; "
                "use fewer complete games/concurrent games or increase memory_budget_mib" %
                (context, self.used, size, self.limit))

    def reserve(self, size: int, context: str) -> None:
        self.check(size, context)
        self.used += size

    def release(self, size: int) -> None:
        self.used -= size


def _model_bytes(model) -> int:
    return sum(t.numel()*t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def _schema_bytes(components, prefix: str, candidates: int = 1) -> int:
    total = 0
    for name, shape in components.schema.shapes.items():
        if name.startswith(prefix):
            total += int(np.prod([candidates if d == "N" else d for d in shape])) * np.dtype(components.schema.dtypes[name]).itemsize
    return total


def _encoded_bytes(prepared: PreparedDecision) -> int:
    obs = prepared.encoded
    arrays = [getattr(obs,f.name) for f in fields(obs) if f.name != "candidates"]
    arrays += [getattr(obs.candidates,f.name) for f in fields(obs.candidates)]
    # Tensor/array object overhead and prepare's retained legal-move string tuple.
    return sum(a.nbytes + 256 for a in arrays) + prepared.memory.numel()*4 + 1024


def _native_bytes(obs) -> int:
    # Native move enums/vectors plus temporary serialized/parsed Python structures,
    # TorchAgent's legal-move strings and the retained native observation. Native
    # allocator capacity isn't exposed; deliberately exceed serialized payload size.
    return 8*len(obs.to_json().encode("utf-8")) + 256*obs.num_legal() + 4096


def _board_workspace(model) -> int:
    """Conservative per-board FP32 budget for stacked input, graph tables and projections."""
    dims = model.schema.dimensions
    entities = ("location", "slot", "link", "merchant", "player")
    rows = sum(dims[name + "s"] for name in entities)
    scalars = dims["global"] + dims["cards"] + sum(dims[name + "s"] * dims[name] for name in entities)
    # A round simultaneously holds input, pooled/next/intermediate and concatenated
    # entity tables. Packed host input and device tensors may coexist.
    return 4 * (3 * scalars + 16 * rows * model.config.entity_dim)


def _workspace(pending, count: int, chunk: int, model) -> int:
    selected = pending[:count]
    candidates = sum(p.observation.num_legal() for p in selected)
    largest = max(p.observation.num_legal() for p in selected)
    e, c, h = model.config.entity_dim, model.config.context_dim, model.config.recurrent_dim
    # The scalar loop may retain the preceding board while creating the next;
    # logits and per-decision vectors span all rows. Optional cache entries yield to this reserve.
    # No-grad candidate embedding/scoring intermediates and full-list logits,
    # masks/probabilities/log-softmax/choice statistics. Chunking never prunes.
    return (2*1024**2 + min(chunk,largest)*(48*e+4*c)*4 + candidates*64
            + min(count, 2)*_board_workspace(model) + count*(h*4 + c*16 + 4096))


def _forward_plan(pending, budget: _MemoryBudget, model) -> Tuple[int, int]:
    count = len(pending)
    while count:
        chunk = 1024
        while True:
            if budget.fits(_workspace(pending,count,chunk,model)):
                return count, chunk
            if chunk == 1:
                break
            chunk = max(1,chunk//2)
        if count == 1:
            budget.check(_workspace(pending,1,1,model),"one complete legal-list forward")
        count = max(1,count//2)
    raise ValueError("cannot forward an empty pending batch")


@dataclass
class _Pending:
    game: object
    seat: int
    observation: brass.Observation
    agent: TorchAgent
    prepared: PreparedDecision
    native_bytes: int
    encoded_bytes: int


class _Game:
    def __init__(self, game_id, lineup, agents, referee, initial, budget, state_bytes, overhead, replay_id):
        self.game_id, self.lineup, self.agents, self.referee = game_id, lineup, agents, referee
        self.initial = initial
        self.replay_id = replay_id
        self.events, self.decisions = [], []
        self.boundaries = {s:[a.memory_snapshot()] for s,a in enumerate(agents) if lineup.entries[s].trainable}
        self.state_bytes, self.overhead = state_bytes, overhead
        budget.reserve(len(initial.encode())+1024 + len(self.boundaries)*state_bytes,"initial episode states")

    def apply(self, index, budget, event_workspace):
        previous = self.initial if not self.events else self.events[-1].public_json
        # Public snapshots grow as discards accumulate; twice the previous JSON
        # plus fixed slack covers event storage and cloned h[k+1] before mutation.
        estimate = 2*len(previous.encode())+16384+len(self.boundaries)*self.state_bytes
        budget.check(estimate+event_workspace,"public event and boundary states")
        event = self.referee.apply_index(index)
        size = len(event.public_json.encode())+len(event.move_json.encode())+512
        budget.reserve(size+len(self.boundaries)*self.state_bytes,"retained public event and h[k+1]")
        self.events.append(event)
        for seat, states in self.boundaries.items():
            states.append(self.agents[seat].memory_snapshot())

    def finish(self, budget):
        episode = Episode(self.game_id,self.initial,tuple(self.events),
                          {s:tuple(states) for s,states in self.boundaries.items()},
                          tuple(self.decisions),self.lineup,self.referee.result(),self.replay_id)
        budget.release(self.overhead)
        return episode


def _frozen_model(template, snapshot: Snapshot, league: League):
    """Task 8 adapts this raw tensor-state-dict loading boundary centrally."""
    league.verify(snapshot)
    frozen = deepcopy(template)
    weights = torch.load(snapshot.weights_path, map_location=template.device, weights_only=True)
    if not isinstance(weights, dict) or not all(isinstance(t, Tensor) for t in weights.values()):
        raise ValueError("snapshot must contain a raw tensor state_dict")
    frozen.load_state_dict(weights, strict=True)
    if not all(torch.isfinite(t).all().item() for t in frozen.state_dict().values()):
        raise ValueError("snapshot has nonfinite weights")
    league.verify(snapshot)
    return frozen.eval().requires_grad_(False)


@torch.no_grad()
@fixed_boards()
def collect_batch(model, league: League, streams: SeedStreams, start_game_id: int,
                  games: int, concurrent_games: int, config: TrainConfig) -> List[Episode]:
    """Finish every requested game with one frozen current-policy copy.

    Failure raises explicitly and returns no partial batch. Retry from the same
    immutable weights/league and start_game_id; seeds never depend on scheduling.
    `boundary_states[seat][k]` has consumed events 1..k, including terminal h[T].
    Only current seats yield Decisions; all moves yield one shared PublicEvent.
    """
    for name,value,minimum in (("start_game_id",start_game_id,0),("games",games,1),
                                ("concurrent_games",concurrent_games,1)):
        if type(value) is not int or value < minimum:
            raise ValueError(name + " must be an integer >= " + str(minimum))
    if start_game_id+games > 2**64:
        raise ValueError("game IDs must fit unsigned 64-bit seed IDs")
    # RuntimeConfig selects a backend; resolved tensor devices may include an index.
    if config.model != model.config or torch.device(model.device).type != config.runtime.device:
        raise ValueError("collection model/config/device mismatch")
    if streams != league.streams:
        raise ValueError("league and collection must use the same controller SeedStreams")
    budget = _MemoryBudget(config.runtime.memory_budget_mib*1024**2)
    weights_bytes = _model_bytes(model)
    # Include caller's current weights, frozen collection copy and controller /
    # component/model Python overhead. This isn't an assertion about process RSS.
    budget.reserve(2*weights_bytes+4*1024**2,"current weights and frozen collection copy")
    components = ComponentData.load()
    current = deepcopy(model).eval().requires_grad_(False)
    budget.reserve(games*1024,"fixed lineups and episode headers")
    lineups = tuple(league.sample(g) for g in range(start_game_id,start_game_id+games))
    snapshots = {s.id:s for s in league.snapshots}
    frozen_models = {}
    for sid in sorted({e.snapshot_id for lineup in lineups for e in lineup.entries if e.kind == "frozen"}):
        budget.check(2*weights_bytes,"frozen weights and loading tensors")
        frozen_models[sid] = _frozen_model(current,snapshots[sid],league)
        budget.reserve(weights_bytes,"retained frozen weights")
    state_bytes = model.config.recurrent_dim*4 + 512
    public_bytes = _schema_bytes(components,"observation.")
    event_workspace = (2*1024**2 + 16*public_bytes + _schema_bytes(components,"candidate.")
                       + 2*_board_workspace(model))
    metadata = recording_metadata()
    from .artifacts import policy_id
    current_id = metadata.pop('candidate_id', None) or policy_id(model, 0)
    history = league.history
    frozen_ids = {sid: replay_identity(sid, row)
                  for sid, row in history.items()}
    metadata.setdefault('category', 'training')
    metadata.setdefault('experiment', experiment_identity(config, streams.master))
    active, completed = [], []
    launched = 0

    def launch():
        nonlocal launched
        game_id = start_game_id+launched
        lineup = lineups[launched]
        # Four private agent states/public parsed snapshots, native game, and
        # transient public event encoding are independent of retained histories.
        neural_seats = sum(e.kind in ("current", "frozen") for e in lineup.entries)
        overhead = 512*1024 + 4*state_bytes + neural_seats*(public_bytes + 6*256)
        budget.reserve(overhead,"active game and independent seat memories")
        agents = []
        for seat,entry in enumerate(lineup.entries):
            seed = streams.seed("policy",game_id,seat)
            if entry.kind in ("current","frozen"):
                policy = current if entry.kind == "current" else frozen_models[entry.snapshot_id]
                agent = TorchAgent(policy,components,seed)
            else:
                agent = brass.RandomBot(seed) if entry.kind == "random" else brass.GreedyBot(seed)
            agents.append(agent)
        budget.check(event_workspace,"initial public memory reset")
        labels = ["current" if e.kind == "current" else (e.snapshot_id or e.kind) for e in lineup.entries]
        identities = [current_id if e.kind == 'current' else
                      frozen_ids[e.snapshot_id] if e.kind == 'frozen' else e.kind for e in lineup.entries]
        recorder = Recorder(seed=streams.seed('train.deal', game_id), seats=labeled_seats(identities),
                            provenance=dict(metadata, game=game_id))
        referee = RefereeGame(streams.seed("train.deal",game_id),agents,labels, recorder=recorder)
        initial = next(a.latest_public_json for a in agents if isinstance(a,TorchAgent))
        game = _Game(game_id,lineup,tuple(agents),referee,initial,budget,state_bytes,overhead,recorder.game_id)
        launched += 1
        return game

    def flush(pending):
        while pending:
            count,chunk = _forward_plan(pending,budget,current)
            batch = pending[:count]
            # Different immutable policies cannot share a forward. Grouping is
            # by model reference only here, never by training-target identity.
            groups = {}
            for item in batch:
                groups.setdefault(item.agent.model,[]).append(item)
            for policy,items in groups.items():
                output = policy([p.prepared.encoded for p in items],
                                torch.cat([p.prepared.memory for p in items]),candidate_chunk=chunk)
                # Complete sampling against this forward, then free its tensors
                # before public-memory encoding allocates another workspace.
                choices = [(item,item.agent.choose(output,row)) for row,item in enumerate(items)]
                del output
                for item,choice in choices:
                    ply = item.prepared.ply
                    item.prepared = None
                    budget.release(item.encoded_bytes)
                    if item.game.lineup.entries[item.seat].trainable:
                        budget.reserve(256,"decision statistics")
                        item.game.decisions.append(Decision(item.game.game_id,ply,item.seat,
                                                           item.observation,choice.index,choice.log_prob,choice.values))
                    item.game.apply(choice.index,budget,event_workspace)
                    if not item.game.lineup.entries[item.seat].trainable:
                        item.observation = None
                        budget.release(item.native_bytes)
            del pending[:count]

    try:
        while launched < games or active:
            while launched < games and len(active) < concurrent_games:
                active.append(launch())
            pending = []
            for game in active:
                seat,obs = game.referee.pending()  # Referee enforces MAX_MOVES.
                agent = game.agents[seat]
                native_bytes = _native_bytes(obs)
                # The next observation is already live, including during an early
                # flush of another game. Charge it before every intervening check.
                budget.reserve(native_bytes,"native observation/candidate list")
                if isinstance(agent,TorchAgent):
                    encoded_estimate = public_bytes+_schema_bytes(components,"candidate.",obs.num_legal())+32768
                    needed = 3*encoded_estimate
                    if pending and not budget.fits(needed+event_workspace):
                        flush(pending)  # All flushed actions were still unsampled.
                    budget.check(needed+event_workspace,"native/encoded complete legal candidates")
                    prepared = agent.prepare(obs)
                    encoded_bytes = _encoded_bytes(prepared)
                    budget.reserve(encoded_bytes,"prepared encoded candidates")
                    pending.append(_Pending(game,seat,obs,agent,prepared,native_bytes,encoded_bytes))
                    del prepared  # Pending owns the only collector reference until choose.
                else:
                    if pending and not budget.fits(event_workspace):
                        flush(pending)
                    game.apply(agent.act(obs),budget,event_workspace)
                    budget.release(native_bytes)
                del obs
            flush(pending)
            remaining = []
            for game in active:
                if game.referee.is_over():
                    completed.append(game.finish(budget))
                else:
                    remaining.append(game)
            active = remaining
    except BaseException:
        for game in active:
            game.referee._recorder.interrupt()
        raise
    return sorted(completed,key=lambda e:e.game_id)
