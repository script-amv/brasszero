# SPDX-License-Identifier: Apache-2.0
"""Real complete-game collection tests; no privileged state enters inference."""
from dataclasses import replace
import json

import brass
import pytest
import torch

from brass_rl.agent import TorchAgent
from brass_rl.config import LeagueConfig, RuntimeConfig, TrainConfig
from brass_rl.features import encode_public
from brass_rl.league import League, Snapshot
from brass_rl.model import BrassPolicy
from brass_rl.rollout import CollectionMemoryError, collect_batch
from brass_rl.schema import ComponentData
from brass_rl.seeding import SeedStreams


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(45)
    components = ComponentData.load()
    config = TrainConfig()
    model = BrassPolicy(config.model, components.schema, components)
    path = tmp_path / "initial.pt"
    torch.save(model.state_dict(), path)
    initial = Snapshot("initial", path, 0, "initial")
    streams = SeedStreams(975318642)
    return model, components, initial, streams, config


def league_for(setup, probabilities):
    _, _, initial, streams, _ = setup
    return League(streams, initial, LeagueConfig(lineup_probabilities=probabilities))


def assert_episode(episode, model, components):
    assert episode.result.moves == len(episode.events) > 100
    assert json.loads(episode.events[-1].public_json)["game_over"]
    assert set(episode.boundary_states) == {s for s,e in enumerate(episode.lineup.entries) if e.trainable}
    previous = episode.initial_public_json
    memories = {s: model.initial_memory(1,"cpu") for s in episode.boundary_states}
    for s in memories:
        assert torch.equal(episode.boundary_states[s][0], memories[s])
        assert len(episode.boundary_states[s]) == len(episode.events)+1
    for k,event in enumerate(episode.events):
        assert event.ply == k+1
        for s in memories:
            with torch.no_grad():
                memories[s] = model.update_memory([encode_public(event,s,previous,components)], memories[s])
            torch.testing.assert_close(episode.boundary_states[s][k+1], memories[s], atol=1e-6, rtol=1e-6)
        previous = event.public_json
    for decision in episode.decisions:
        assert decision.key == (episode.game_id, decision.ply, decision.seat)
        assert isinstance(decision.observation, brass.Observation)
        assert episode.lineup.entries[decision.seat].trainable
        assert 0 <= decision.index < decision.observation.num_legal()
        assert decision.observation.legal_moves()[decision.index] == episode.events[decision.ply].move_json
        assert decision.seat == episode.events[decision.ply].actor
        assert decision.observation.public_json() == (episode.initial_public_json if decision.ply == 0 else episode.events[decision.ply-1].public_json)
        with torch.no_grad():
            from brass_rl.features import encode_observation
            from brass_rl.distributions import policy_stats
            output = model([encode_observation(decision.observation,components)],episode.boundary_states[decision.seat][decision.ply])
            stats = policy_stats(output,torch.tensor([decision.index]))
        assert decision.old_log_prob == pytest.approx(stats.log_prob.item(),abs=2e-5)
        assert decision.old_values == pytest.approx(stats.values[0].tolist(),abs=1e-6)


def test_collection_freezes_weights_and_lineups(setup, monkeypatch):
    model, components, initial, streams, config = setup
    before = {k:t.clone() for k,t in model.state_dict().items()}
    old_file = initial.weights_path.read_bytes()
    global_rng = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *args, **kwargs: pytest.fail("optimizer during collection"))
    episodes = collect_batch(model,league_for(setup,(1,0,0)),streams,0,2,2,config)
    assert [e.game_id for e in episodes] == [0,1]
    from brass_ai.replay_archive import Archive
    records = {r['provenance']['game']: r for r in Archive().records()}
    assert set(records) == {0, 1}
    for episode in episodes:
        record = records[episode.game_id]
        assert record['result']['vp'] == list(episode.result.vp)
        assert record['result']['ranking'] == list(episode.result.ranking)
        assert record['moves'] == [{'player': e.actor, 'move': json.loads(e.move_json)} for e in episode.events]
        assert all(s['bot'].startswith('policy-sha256-v1:') for s in record['seats'])
    assert initial.weights_path.read_bytes() == old_file
    assert all(torch.equal(before[k],t) for k,t in model.state_dict().items())
    assert model.training
    assert torch.equal(global_rng,torch.random.get_rng_state())
    for episode in episodes:
        assert {d.seat for d in episode.decisions} == {0,1,2,3}
        assert len(episode.decisions) == len(episode.events)
        assert_episode(episode,model,components)
    # Separate current/frozen weights prove that the collector actually loads
    # the immutable artifact rather than reusing today's current parameters.
    old_weight = next(model.parameters()).flatten()[0].item()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    seen_weights = set()
    forward = BrassPolicy.forward
    def frozen_forward(self, observations, memory, candidate_chunk=1024):
        assert not torch.is_grad_enabled()
        assert not any(p.requires_grad for p in self.parameters())
        assert self is not model
        seen_weights.add(next(self.parameters()).flatten()[0].item())
        return forward(self,observations,memory,candidate_chunk)
    with monkeypatch.context() as scoped:
        scoped.setattr(BrassPolicy,"forward",frozen_forward)
        frozen = collect_batch(model,league_for(setup,(0,1,0)),streams,2,1,1,config)[0]
    assert seen_weights == {old_weight,next(model.parameters()).flatten()[0].item()}
    assert initial.weights_path.read_bytes() == old_file
    assert len({d.seat for d in frozen.decisions}) == 1
    assert len(frozen.decisions) < len(frozen.events)
    assert_episode(frozen,model,components)


def test_interleaved_games_isolate_memory(setup, monkeypatch):
    model, components, _, streams, config = setup
    seen = []
    original = TorchAgent.__init__
    def record(self, m, c, seed, *args, **kwargs):
        seen.append(seed)
        return original(self,m,c,seed,*args,**kwargs)
    monkeypatch.setattr(TorchAgent,"__init__",record)
    league = league_for(setup,(1,0,0))
    first = collect_batch(model,league,streams,10,2,1,config)
    seeds = tuple(seen)
    seen.clear()
    from brass_ai.replay_archive import Archive
    original_ids = {r['game_id'] for r in Archive().records()}
    operational_override = replace(config, runtime=replace(config.runtime, max_wall_time_seconds=999,
                                                          checkpoint_margin_seconds=1))
    second = collect_batch(model,league,streams,10,2,2,operational_override)
    assert {r['game_id'] for r in Archive().records()} == original_ids
    assert sorted(seeds) == sorted(seen)
    assert len(set(seeds)) == 8
    assert set(seeds).isdisjoint({streams.master, *(e.result.seed for e in first)})
    assert seeds == tuple(streams.seed("policy", g,s) for g in (10,11) for s in range(4))
    for a,b in zip(first,second):
        assert a.lineup == b.lineup
        assert a.result == b.result
        assert a.events == b.events
        assert [d.key for d in a.decisions] == [d.key for d in b.decisions]
        for s in range(4):
            torch.testing.assert_close(torch.cat(a.boundary_states[s]),torch.cat(b.boundary_states[s]),atol=2e-6,rtol=2e-6)


def test_memory_and_move_bounds_fail_clearly(setup, monkeypatch):
    model, _, _, streams, config = setup
    tiny = replace(config,runtime=RuntimeConfig(memory_budget_mib=1))
    with pytest.raises(CollectionMemoryError,match="memory|budget"):
        collect_batch(model,league_for(setup,(1,0,0)),streams,0,1,1,tiny)
    monkeypatch.setattr("brass_ai.arena.MAX_MOVES",1)
    with pytest.raises(RuntimeError,match="exceeded.*moves"):
        collect_batch(model,league_for(setup,(1,0,0)),streams,0,1,1,config)


def test_rejects_bad_collection_counts(setup):
    model, _, _, streams, config = setup
    for arguments in ((-1,1,1),(0,0,1),(0,1,0),(0,True,1)):
        with pytest.raises(ValueError):
            collect_batch(model,league_for(setup,(1,0,0)),streams,*arguments,config)


def test_pending_plan_reduces_batches_before_sampling(setup):
    from types import SimpleNamespace
    from brass_rl.rollout import _MemoryBudget, _forward_plan
    model = setup[0]
    class LargeLegalList:
        def num_legal(self):
            return 100000
    pending = [SimpleNamespace(observation=LargeLegalList()) for _ in range(4)]
    original = list(pending)
    count,chunk = _forward_plan(pending,_MemoryBudget(9*1024**2),model)
    assert count == 1
    assert 1 <= chunk < 1024
    assert pending == original  # Planning neither chooses nor discards a game.
    with pytest.raises(CollectionMemoryError,match="complete legal-list"):
        _forward_plan(pending,_MemoryBudget(1024),model)


def test_episode_allocation_budget_fails_without_returning_partial_games(setup, monkeypatch):
    model, _, _, streams, config = setup
    choices = []
    original = TorchAgent.choose
    def choose(self, output, row):
        choice = original(self,output,row)
        choices.append(choice)
        return choice
    monkeypatch.setattr(TorchAgent,"choose",choose)
    config = replace(config,runtime=RuntimeConfig(memory_budget_mib=20))
    with pytest.raises(CollectionMemoryError,match="retained=.*budget="):
        collect_batch(model,league_for(setup,(1,0,0)),streams,0,2,2,config)
    assert choices  # Candidate/history growth is checked after initial allocation.


def test_evicted_snapshot_files_can_be_pruned(setup, tmp_path):
    model, _, initial, streams, config = setup
    league = League(streams,initial,LeagueConfig(max_snapshots=3))
    for update in (1,2,3):
        path = tmp_path / ("snapshot%d.pt" % update)
        torch.save(model.state_dict(),path)
        league.add(Snapshot("s%d" % update,path,update,"training"))
    (tmp_path / "snapshot2.pt").unlink()
    assert {s.id for s in league.snapshots} == {"initial","s1","s3"}
    assert len(league.sample(8).entries) == 4


def test_mixed_baselines_and_hooks_receive_only_separated_agent_seeds(setup, monkeypatch):
    from brass_rl.league import Lineup, LineupEntry
    model, _, initial, streams, config = setup
    league = league_for(setup,(0,0,1))
    lineup = Lineup((LineupEntry("current",trainable=True),LineupEntry("frozen",initial.id),
                     LineupEntry("random"),LineupEntry("greedy")))
    monkeypatch.setattr(league,"sample",lambda game_id: lineup)
    received = []
    random_bot, greedy_bot = brass.RandomBot, brass.GreedyBot
    def baseline(factory,seat):
        def create(seed):
            assert seed == streams.seed("policy",30,seat)
            received.append(seed)
            return factory(seed)
        return create
    monkeypatch.setattr(brass,"RandomBot",baseline(random_bot,2))
    monkeypatch.setattr(brass,"GreedyBot",baseline(greedy_bot,3))
    reset = TorchAgent.reset
    observe = TorchAgent.observe_public
    def reset_public(self,seat,initial_public_json):
        assert type(seat) is int and seat in range(4)
        public = json.loads(initial_public_json)
        assert not {"seed","master","deck","hands","my_hand"}.intersection(public)
        return reset(self,seat,initial_public_json)
    def observe_public(self,event):
        assert set(vars(event)) == {"ply","actor","move_json","public_json"}
        assert not {"seed","master","deck","hands","my_hand"}.intersection(json.loads(event.public_json))
        return observe(self,event)
    monkeypatch.setattr(TorchAgent,"reset",reset_public)
    monkeypatch.setattr(TorchAgent,"observe_public",observe_public)
    episode = collect_batch(model,league,streams,30,1,1,config)[0]
    assert received == [streams.seed("policy",30,2),streams.seed("policy",30,3)]
    assert set(received).isdisjoint({streams.master,episode.result.seed})
    assert {d.seat for d in episode.decisions} == {0}
    assert {e.actor for e in episode.events} == {0,1,2,3}
    assert len(episode.boundary_states[0]) == episode.result.moves+1
    assert json.loads(episode.events[-1].public_json)["game_over"]


def test_prepared_candidates_are_freed_before_event_and_replacement(setup, monkeypatch):
    import weakref
    import brass_rl.rollout as rollout
    model, _, _, streams, config = setup
    references = []
    applied = []
    prepare, apply = TorchAgent.prepare, rollout._Game.apply
    class CheckedTwoEvents(Exception):
        pass
    def tracked_prepare(self,obs):
        # One active game: its previous encoded candidates must have died before
        # encoding this replacement, not just after the replacement is allocated.
        assert all(ref() is None for refs in references for ref in refs)
        prepared = prepare(self,obs)
        references.append((weakref.ref(prepared),weakref.ref(prepared.encoded.candidates.features)))
        return prepared
    def checked_apply(self,index,budget,event_workspace):
        assert all(agent._prepared is None for agent in self.agents)
        assert all(ref() is None for refs in references for ref in refs), "released prepared candidates are still live during event processing"
        apply(self,index,budget,event_workspace)
        applied.append(index)
        if len(applied) == 2:
            raise CheckedTwoEvents
    monkeypatch.setattr(TorchAgent,"prepare",tracked_prepare)
    monkeypatch.setattr(rollout._Game,"apply",checked_apply)
    with pytest.raises(CheckedTwoEvents):
        collect_batch(model,league_for(setup,(1,0,0)),streams,0,1,1,config)
    assert len(references) == len(applied) == 2


@pytest.mark.parametrize("waiting_kind",["current","random"])
def test_waiting_native_observation_is_charged_during_early_flush(setup, monkeypatch, waiting_kind):
    import brass_rl.rollout as rollout
    from brass_rl.league import Lineup, LineupEntry
    model, components, _, streams, config = setup
    league = league_for(setup,(1,0,0))
    current = LineupEntry("current",trainable=True)
    baseline_actor = brass.Game(streams.seed("train.deal",1)).current_player()
    focal = (baseline_actor+1)%4
    baseline = Lineup(tuple(current if s == focal else LineupEntry("random") for s in range(4)))
    monkeypatch.setattr(league,"sample",lambda game_id: baseline if game_id == 1 and waiting_kind == "random" else Lineup((current,)*4))
    budgets, receipts, waiting = [], [], []
    base_budget, native_bytes = rollout._MemoryBudget, rollout._native_bytes
    public_bytes = rollout._schema_bytes(components,"observation.")
    event_workspace = 2*1024**2+16*public_bytes+rollout._schema_bytes(components,"candidate.")
    class MeasuredBudget(base_budget):
        def __init__(self,limit):
            super().__init__(limit)
            budgets.append(self)
    class CheckedWaitingAllocation(Exception):
        pass
    def receive(obs):
        size = native_bytes(obs)
        receipts.append(size)
        if len(receipts) == 2:
            budget = budgets[0]
            # Put the real allocation immediately below the threshold needed to
            # prepare the second game, so the first game's pending decision must
            # flush while this newly received native observation is still live.
            extra = 0
            if waiting_kind == "current":
                encoded = public_bytes+rollout._schema_bytes(components,"candidate.",obs.num_legal())+32768
                extra = 3*encoded
            waiting.append(budget.used+size)
            budget.limit = budget.used+size+extra+event_workspace-1
        return size
    def checked_plan(pending,budget,policy):
        assert len(receipts) == 2 and len(pending) == 1
        assert budget.used == waiting[0], "planner omitted the live waiting native observation"
        raise CheckedWaitingAllocation
    monkeypatch.setattr(rollout,"_MemoryBudget",MeasuredBudget)
    monkeypatch.setattr(rollout,"_native_bytes",receive)
    monkeypatch.setattr(rollout,"_forward_plan",checked_plan)
    with pytest.raises(CheckedWaitingAllocation):
        collect_batch(model,league,streams,0,2,2,config)


@pytest.mark.parametrize('actual,configured,compatible', [
    ('cpu', 'cpu', True), ('cpu:0', 'cpu', True),
    ('cuda', 'cuda', True), ('cuda:0', 'cuda', True), ('cuda:3', 'cuda', True),
    ('mps', 'mps', True), ('mps:0', 'mps', True),
    ('cuda:0', 'cpu', False), ('mps:0', 'cuda', False), ('cpu', 'mps', False),
])
def test_collection_device_backend_contract_without_accelerator_allocation(actual, configured, compatible):
    """Config names a backend; resolved tensor indices must not reject that backend.

    Stop at the next real validation guard (different controller seed streams),
    before any model copy/forward/allocation. These are real torch.device objects,
    not a claim that accelerator training ran on this CPU-only machine.
    """
    from types import SimpleNamespace
    config = TrainConfig(runtime=RuntimeConfig(device=configured))
    model = SimpleNamespace(config=config.model, device=torch.device(actual))
    streams = SeedStreams(11)
    league = SimpleNamespace(streams=SeedStreams(12))
    message = 'league and collection' if compatible else 'model/config/device mismatch'
    with pytest.raises(ValueError, match=message):
        collect_batch(model, league, streams, 0, 1, 1, config)


def test_scalar_forward_admits_live_boards_and_all_retained_rows_before_allocation(setup):
    from types import SimpleNamespace
    from brass_rl.rollout import _board_workspace, _forward_plan, _MemoryBudget, _workspace
    model = setup[0]
    item = SimpleNamespace(observation=SimpleNamespace(num_legal=lambda: 1))
    pending = [item] * 4
    required = _workspace(pending, 4, 1, model)
    assert required - _workspace(pending, 1, 1, model) >= _board_workspace(model)
    count, _ = _forward_plan(pending, _MemoryBudget(required - 1), model)
    assert 1 <= count < 4 and len(pending) == 4
