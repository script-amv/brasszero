# SPDX-License-Identifier: Apache-2.0
from collections import Counter

from brass_rl.league_study import schedule, summarize


def test_schedule_and_deal_block_differences():
    rows = schedule()
    assert len(rows)==50 and len({r['deal_seed'] for r in rows})==25
    assert sorted(Counter(r['candidate_seat'] for r in rows).values())==[12,12,13,13]
    assert all(n==2 for n in Counter(r['deal_id'] for r in rows).values())
    arms = {}
    for name, increment in [('old',0),('new',3)]:
        arms[name] = [dict(r, vp=[10+increment]*4, ranking=[0,1,2,3],
                           seconds=.2, replay_id=name+str(i)) for i,r in enumerate(rows)]
    metrics = summarize(arms)
    assert metrics['arms']['new']['vp']['q25']==13
    assert metrics['paired']['mean_vp_delta']==3
    assert len(metrics['paired']['deal_blocks'])==25


def test_recovery_budget_rejects_repeats_and_early_evaluation():
    from brass_rl.league_study import check_recovery_budget, BASE, END
    import pytest
    entry = dict(update=BASE,manifests={'replay_ids':[]})
    game = lambda i: dict(game_id=i,deal_seed=1000+i,outcome=dict(ranking=[0,1,2,3],vp=[4,3,2,1]))
    update = dict(update=BASE+1,manifests=dict(replay_ids=['a','b'],train=[game(1586),game(1587)]))
    journals = {'old':[entry,update],'new':[entry]}
    ids = {'old':'x','new':'y'}
    records = [dict(game_id=k,status='completed',category='training',provenance={'experiment':'x','game':i,'update':BASE+1},
                    initial={'seed':1000+i},result=game(i)['outcome']) for k,i in [('a',1586),('b',1587)]]
    assert check_recovery_budget(journals,ids,records,{})=={'old':BASE+1,'new':BASE}
    with pytest.raises(ValueError,match='uncommitted'):
        check_recovery_budget(journals,ids,records+[dict(records[0],game_id='uncommitted',provenance={'experiment':'x','game':1588,'update':BASE+2})],{})
    with pytest.raises(ValueError,match='early evaluation'):
        check_recovery_budget(journals,ids,records+[dict(records[0],game_id='extra',category='evaluation')],{})
    over = dict(journals,old=[entry,dict(update=END+1,manifests={})])
    with pytest.raises(ValueError,match='budget'):
        check_recovery_budget(over,ids,records,{})
    with pytest.raises(ValueError,match='missing'):
        check_recovery_budget(journals,ids,records[:1],{})


def test_treatment_verifier_rejects_swapped_opponents_or_rated_game():
    from copy import deepcopy
    import pytest
    from brass_ai.replay_archive import labeled_seats
    from brass_rl.league_study import check_dynamic_treatment
    from brass_rl.seeding import SeedStreams
    master = 321
    seat = {'kind':'current','snapshot_id':None,'trainable':True}
    peers = ['a','b','c']
    games, records, rated = [], {}, []
    for game_id in (1586,1587):
        lineup = [seat]*4
        if game_id%2!=SeedStreams(master).seed('dynamic.self_play',game_id//2)%2:
            lineup = [seat]+[dict(kind='frozen',snapshot_id=p,trainable=False) for p in peers]
            rated.append(game_id)
        games.append(dict(game_id=game_id,lineup=lineup))
        records[game_id] = dict(seats=labeled_seats([e['snapshot_id'] or 'learner' for e in lineup]))
    check_dynamic_treatment(games,records,{793:'learner'},peers,rated,master)
    with pytest.raises(ValueError,match='rated'):
        check_dynamic_treatment(games,records,{793:'learner'},peers,[3173-rated[0]],master)
    changed = deepcopy(records)
    changed[rated[0]]['seats'][1]['bot']='wrong-peer'
    with pytest.raises(ValueError,match='seat'):
        check_dynamic_treatment(games,changed,{793:'learner'},peers,rated,master)


def test_tip_cursor_and_master_must_match_accepted_boundary():
    from types import SimpleNamespace
    import pytest
    from brass_rl.league_study import check_tip
    cfg = SimpleNamespace(master_seed=1)
    state = SimpleNamespace(config=cfg,master_seed=1,completed_updates=794,next_game_id=1588,dynamic_league={})
    check_tip('new',state,794,cfg)
    state.next_game_id -= 2
    with pytest.raises(ValueError,match='cursor'):
        check_tip('new',state,794,cfg)
    state.next_game_id=1588
    state.master_seed=2
    with pytest.raises(ValueError,match='master'):
        check_tip('new',state,794,cfg)


def test_correct_replay_id_does_not_authorize_wrong_saved_evaluation(tmp_path,monkeypatch):
    from brass_ai.replay_archive import Archive,digest,engine_id,labeled_seats
    from brass_rl import combination_evaluation
    from brass_rl.learning_study import write
    from brass_rl.league_study import check_evaluation_row,evaluation_identity
    import pytest
    plan = schedule()[0]
    lineage,origin = evaluation_identity(plan,'endpoint','weights','experiment')
    replay_id = digest(dict(suite=lineage['suite_id'],origin='experiment',game=0,policies=lineage['seat_ids']))
    archive=Archive(tmp_path/'replays')
    actual = [44,33,22,11]
    ranks = [0,1,2,3]
    rec=dict(game_id=replay_id,status='completed',engine_id=engine_id(),seats=labeled_seats(lineage['labels']),
             initial={'seed':plan['deal_seed']},provenance=origin,moves=[],result={'vp':actual,'ranking':ranks})
    archive.publish(rec)
    # Isolate the row-to-record check. Production separately replays every native move.
    monkeypatch.setattr(combination_evaluation,'validate_replay',lambda r:None)
    path=tmp_path/'eval/games/0000.json'
    row=dict(lineage,replay_id=replay_id,seconds=.3,ranking=ranks,vp=actual,plies=0,terminal_brier=None)
    write(path,row)
    check_evaluation_row(archive,path,lineage,origin,replay_id)
    write(path,dict(row,vp=[99,33,22,11]))
    with pytest.raises(ValueError,match='differs from its audited replay'):
        check_evaluation_row(archive,path,lineage,origin,replay_id)


def test_audit_reconstructs_weighted_draws_and_each_boundary_rating(tmp_path,monkeypatch):
    from copy import deepcopy
    from dataclasses import asdict
    from types import SimpleNamespace
    from brass_rl import artifacts, league_study as study
    from brass_rl.dynamic_league import DynamicLeague
    from brass_rl.seeding import SeedStreams
    from test_dynamic_league import snapshot
    import pytest
    members = [snapshot(tmp_path,n) for n in range(0,794,20)]
    pool = DynamicLeague.start(SeedStreams(12), members, (), history_extent=794,next_game_id=1586)
    def state():
        return SimpleNamespace(dynamic_league=pool.state,league_history=pool.history,
                               league_snapshot_ids=tuple(m.id for m in members),holdouts=(),master_seed=12)
    states = {793:state()}
    journal = [dict(update=793,manifests={})]
    for update in (794,795):
        games=[]
        for g in (2*update-2,2*update-1):
            lineup=pool.sample(g)
            rank=[0,1,2,3]
            games.append(dict(game_id=g,lineup=[asdict(e) for e in lineup.entries],outcome=dict(ranking=rank)))
            pool.record_training(g,lineup,rank)
        states[update]=state()
        journal.append(dict(update=update,manifests=dict(train=games)))
    monkeypatch.setattr(artifacts,'inspect_update_state',lambda path:states[int(path.name)])
    study.check_dynamic_history(tmp_path,journal,members,states[795])
    changed=deepcopy(journal)
    historical=next(g for g in changed[1]['manifests']['train'] if any(e['kind']=='frozen' for e in g['lineup']))
    seats=[e for e in historical['lineup'] if e['kind']=='frozen']
    seats[0]['snapshot_id'],seats[1]['snapshot_id']=seats[1]['snapshot_id'],seats[0]['snapshot_id']
    with pytest.raises(ValueError,match='sampling history'):
        study.check_dynamic_history(tmp_path,changed,members,states[795])
    states[794].dynamic_league['ratings'][members[0].id]['mu']+=.1
    with pytest.raises(ValueError,match='training results'):
        study.check_dynamic_history(tmp_path,journal,members,states[795])
