# SPDX-License-Identifier: Apache-2.0
import hashlib
from dataclasses import replace

import pytest

from brass_rl import dynamic_league as dl
from brass_rl.league import Snapshot
from brass_rl.seeding import SeedStreams


def snapshot(root, update, variant='canonical'):
    data = (variant+str(update)).encode()
    path = root/(variant+str(update))
    path.write_bytes(data)
    return Snapshot('policy-sha256-v1:'+hashlib.sha256(data).hexdigest(), path, update,
                    'initial' if update == 0 else 'training')


def league(root):
    members = [snapshot(root, n) for n in range(0, 81, 20)]
    return dl.DynamicLeague.start(SeedStreams(45), members, (), history_extent=81, next_game_id=162)


def test_ancestry_and_linux_original():
    def row(n, branch, tag, lineage='mainline'):
        return dict(update=n, branch=branch, policy_id=tag, lineage=lineage)
    rows = [row(260,'school','re-export'), row(260,'school-original','linux'),
            row(700,'control','main-control'), row(700,'ABC','foreign','experimental'),
            row(740,'BC','main-bc'), row(740,'control','parallel','experimental'),
            row(800,'BC','future'), row(800,'study','experiment','experimental')]
    assert [r['policy_id'] for r in dl.mainline_rows(rows, 800)] == [
        'linux','main-control','main-bc','future']
    with pytest.raises(ValueError, match='ambiguous'):
        dl.mainline_rows(rows+[row(740,'BC','other')], 800)


def test_unique_seats_ratings_resume_and_thinning(tmp_path):
    pool = league(tmp_path)
    first, second = [pool.sample(g) for g in (162,163)]
    history = next(l for l in (first,second) if sum(s.trainable for s in l.entries)==1)
    assert sum(s.trainable for l in (first,second) for s in l.entries)==5
    seats = [s.snapshot_id for s in history.entries if s.kind=='frozen']
    assert len(set(seats)) == 3
    for game_id, lineup in enumerate((first,second),162):
        pool.record_training(game_id, lineup, (0,1,1,3))
    state = pool.state
    assert len(state['rated_games']) == 1
    assert all(state['ratings'][sid]['sigma'] < 25/3 for sid in seats)
    restored = dl.DynamicLeague.restore(pool.streams, pool.snapshots, pool.holdouts, pool.history, state)
    assert restored.sample(165) == pool.sample(165)
    with pytest.raises(ValueError,match='next'):
        restored.record_training(163,second,(0,1,1,3))
    for update in range(81,162):
        restored.advance_main(update, snapshot(tmp_path,update) if update % restored.stride==0 else None)
    assert restored.stride == 40
    assert restored.state['next_thinning_extent']==324
    assert [s.update for s in restored.snapshots]==[0,40,80,120,160]
    for s in restored.snapshots:
        if s.id in state['ratings']:
            assert state['ratings'][s.id] == restored.state['ratings'][s.id]
    for update in range(162,324):
        restored.advance_main(update, snapshot(tmp_path,update) if update % restored.stride==0 else None)
    assert restored.stride==80 and [s.update for s in restored.snapshots]==[0,80,160,240,320]


def test_migrated_full_state_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from test_train import tiny_config, tiny_collection, assert_nested_equal
    from brass_rl import train as controller
    from brass_rl.artifacts import load_update, save_update, restore_rng_states
    from brass_rl.policy_io import policy_metadata, tensor_bytes
    import torch
    from dataclasses import replace
    monkeypatch.setattr(controller, 'collect_batch', tiny_collection)
    monkeypatch.setattr(controller, 'retain_training_replays', lambda *args: ())
    # This short PPO fixture has a real decision at seat zero only. Choose a
    # seed putting that seat in both scheduled historical lineups.
    import random
    for master in range(1000):
        streams = SeedStreams(master)
        focal_games = [g for g in range(80,84) if g%2 != streams.seed('dynamic.self_play',g//2)%2]
        if all(random.Random(streams.seed('opponents',g)).randrange(4)==0 for g in focal_games):
            break
    cfg = replace(tiny_config(),master_seed=master)
    initial = controller.train(cfg,tmp_path/'source',max_updates=0)
    loaded = load_update(initial.latest_checkpoint)
    source = tmp_path/'members'
    source.mkdir()
    members = []
    for n in (0,20,40):
        path = source/str(n)
        path.write_bytes(tensor_bytes(loaded.model.state_dict()))
        identity = policy_metadata(loaded.model,n,hashlib.sha256(path.read_bytes()).hexdigest())['id']
        members.append(Snapshot(identity,path,n,'initial' if n==0 else 'training'))
    for name in ('split','full'):
        run = tmp_path/name
        optimizer = torch.optim.Adam(loaded.model.parameters(),lr=cfg.ppo.learning_rate)
        restore_rng_states(loaded.state.rng_states)
        save_update(run,loaded.model,optimizer,replace(loaded.state,completed_updates=40,next_game_id=80))
    full = controller.train(cfg,tmp_path/'full',tmp_path/'full/latest',max_updates=42,dynamic_members=members)
    controller.train(cfg,tmp_path/'split',tmp_path/'split/latest',max_updates=41,dynamic_members=members)
    split = controller.train(cfg,tmp_path/'split',tmp_path/'split/latest',max_updates=42)
    assert full.state.dynamic_league == split.state.dynamic_league
    assert len(full.state.dynamic_league['rated_games'])==2
    assert full.state.dynamic_league['history_extent']==41
    assert full.choice_trace == split.choice_trace
    assert_nested_equal(load_update(split.latest_checkpoint).optimizer_state,
                        load_update(full.latest_checkpoint).optimizer_state)
    assert load_update(initial.latest_checkpoint).state.dynamic_league is None


@pytest.fixture
def activation_run(tmp_path, monkeypatch):
    """A private 793 controller and 40 valid policy IDs with tiny QA weights."""
    from brass_rl import main_bc, train as controller
    from brass_rl.artifacts import load_update, save_update
    from brass_rl.policy_io import policy_metadata, tensor_bytes
    from test_train import tiny_config
    import torch
    import random
    import json
    from dataclasses import asdict
    for master in range(100):
        streams = SeedStreams(master)
        game = next(g for g in (1586,1587) if g%2 != streams.seed('dynamic.self_play',g//2)%2)
        if random.Random(streams.seed('opponents',game)).randrange(4)==0:
            break
    cfg = replace(tiny_config(), master_seed=master)
    (tmp_path/'profile.json').write_text(json.dumps(asdict(cfg)))
    monkeypatch.setattr(main_bc, 'ROOT', tmp_path)
    monkeypatch.setattr(main_bc, 'PROFILE', 'profile.json')
    monkeypatch.setattr(main_bc, 'prepare_main_run', lambda root: root/'run')
    source = controller.train(cfg,tmp_path/'source',max_updates=0)
    loaded = load_update(source.latest_checkpoint)
    weights = tmp_path/'fixture-weights.pt'
    weights.write_bytes(tensor_bytes(loaded.model.state_dict()))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    members = [Snapshot(policy_metadata(loaded.model,n,digest)['id'],weights,n,
                        'training' if n else 'initial') for n in range(0,794,20)]
    monkeypatch.setattr(dl,'catalog_members',lambda project,tip: members)
    run = tmp_path/'run'
    base = save_update(run, loaded.model, torch.optim.Adam(loaded.model.parameters(),lr=cfg.ppo.learning_rate),
                       replace(loaded.state, completed_updates=793, next_game_id=1586))
    (run/'config.json').write_bytes((tmp_path/'profile.json').read_bytes())
    return main_bc, controller, run, base, weights


def qa_recorders(league, game):
    from brass_ai.replay_archive import Recorder, labeled_seats, recording_metadata
    from brass_rl.league import replay_identity
    meta = recording_metadata()
    candidate = meta.pop('candidate_id')
    lineup = league.sample(game)
    labels = [candidate if e.trainable else replay_identity(e.snapshot_id,league.history[e.snapshot_id])
              if e.kind=='frozen' else e.kind for e in lineup.entries]
    return Recorder(seed=league.streams.seed('train.deal',game),seats=labeled_seats(labels),
                    provenance=dict(meta,game=game))


def test_first_activation_survives_inflight_crash_and_ordinary_resume(activation_run,monkeypatch):
    from brass_ai.replay_archive import Archive
    from brass_rl.artifacts import load_update
    from test_train import tiny_collection
    from types import SimpleNamespace
    main, controller, run, base, weights = activation_run
    original = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
    calls, expected = [], {}
    def collect(model,league,streams,start,games,concurrent,config):
        ids = []
        preview = dl.DynamicLeague.restore(streams,league.snapshots,league.holdouts,league.history,league.state)
        for g in range(start,start+games):
            recorder = qa_recorders(league,g)
            ids.append(recorder.game_id)
            recorder.append(0,'fixture action')
            ranks = [0,1,2,3] if g==start else [1,0,2,3]
            preview.record_training(g,preview.sample(g),ranks)
            if not calls:
                recorder.interrupt()
            else:
                recorder.finish(SimpleNamespace(is_over=lambda:True,ranking=lambda:ranks,vp=lambda:[4,3,2,1]))
        calls.append(ids)
        if len(calls)==1:
            expected.update(preview.state)
            raise RuntimeError('simulated loss before first PPO')
        return tiny_collection(model,league,streams,start,games,concurrent,config)
    monkeypatch.setattr(controller,'collect_batch',collect)
    monkeypatch.setattr(controller,'retain_training_replays',lambda *args: ())
    preview = main.continue_bc(793,lambda:False,dynamic=True)
    assert len(preview.preflight['opponents']['training'])==40
    assert len(main.continue_bc(793,lambda:False).preflight['opponents']['training'])==1
    with pytest.raises(RuntimeError,match='simulated loss'):
        main.continue_bc(794,lambda:False,dynamic=True)
    assert load_update(run/'latest').state.dynamic_league is None
    assert len(Archive().records())==2
    weights.unlink()  # Ordinary recovery uses the run's sealed content-addressed members.
    resumed = main.continue_bc(794,lambda:False)
    assert calls[0]==calls[1]
    assert len(resumed.preflight['opponents']['training'])==40
    assert resumed.state.dynamic_league['rated_games']==expected['rated_games']
    assert len(expected['rated_games'])==1
    assert resumed.state.dynamic_league['ratings']==expected['ratings']
    assert sorted(resumed.state.manifests['replay_ids'])==sorted(calls[0])
    assert [r['status'] for r in Archive().records()]==['completed']*2
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}==original


def test_pending_legacy_game_blocks_first_activation(activation_run,monkeypatch):
    from brass_ai.replay_archive import Archive
    main,controller,run,base,weights = activation_run
    def interrupted(model,league,streams,start,*args):
        qa_recorders(league,start).interrupt()
        raise RuntimeError('legacy prefix')
    monkeypatch.setattr(controller,'collect_batch',interrupted)
    with pytest.raises(RuntimeError,match='legacy prefix'):
        main.continue_bc(794,lambda:False)
    before = [r['game_id'] for r in Archive().records()]
    monkeypatch.setattr(controller,'collect_batch',lambda *args:pytest.fail('started conflicting collection'))
    with pytest.raises(ValueError,match='pending'):
        main.continue_bc(794,lambda:False,dynamic=True)
    assert [r['game_id'] for r in Archive().records()]==before
    assert len(main.continue_bc(793,lambda:False).preflight['opponents']['training'])==1


def test_conflicting_pending_identity_refused_on_unflagged_recovery(activation_run,monkeypatch):
    from brass_ai.replay_archive import Archive
    main, controller, run, base, weights = activation_run
    def interrupted(model,league,streams,start,*args):
        recorder = qa_recorders(league,start)
        recorder.interrupt()
        raise RuntimeError('dynamic prefix')
    monkeypatch.setattr(controller,'collect_batch',interrupted)
    with pytest.raises(RuntimeError,match='dynamic prefix'):
        main.continue_bc(794,lambda:False,dynamic=True)
    archive = Archive()
    record, = archive.records()
    # Retain the actual pending prefix and inject a second ID for the same game.
    archive.publish(dict(record,game_id='a'*64))
    before = {p.name:p.read_bytes() for p in archive.games.iterdir()}
    monkeypatch.setattr(controller,'collect_batch',lambda *args:pytest.fail('started conflicting collection'))
    with pytest.raises(ValueError,match='pending training recording conflicts'):
        main.continue_bc(794,lambda:False)
    assert {p.name:p.read_bytes() for p in archive.games.iterdir()}==before
