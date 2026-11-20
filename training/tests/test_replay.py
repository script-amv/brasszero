# SPDX-License-Identifier: Apache-2.0
import json
import brass
import pytest
from brass_ai.arena import RefereeGame
from brass_rl.replay import write_gui_replay, select_replays, retain_training_replays


def test_replay_reconstructs_ranks_vp_and_large_seed(tmp_path):
    seed=2**63+291
    agents=[brass.RandomBot(i) for i in range(4)]
    referee=RefereeGame(seed,agents,['a','b','b','d'])
    events=[]
    while not referee.is_over():
        seat,obs=referee.pending()
        events.append(referee.apply_index(agents[seat].act(obs)))
    path=tmp_path/'game.json'
    write_gui_replay(path,seed,['a','b','b','d'],events)
    payload=json.loads(path.read_text())
    assert payload['seed'] == seed and payload['format'] == 'brass-gui-game/1'
    assert payload['seats'] == [dict(kind='human',name=n,bot=None) for n in ['a','b','b','d']]
    replay=brass.Game(payload['seed'])
    for record in payload['moves']:
        assert replay.current_player() == record['player']
        obs=replay.observation(record['player'])
        index=[json.loads(m) for m in obs.legal_moves()].index(record['move'])
        replay.apply_index(index)
    assert tuple(replay.ranking()) == referee.result().ranking
    assert tuple(replay.vp()) == referee.result().vp
    with pytest.raises(ValueError,match='complete'):
        write_gui_replay(tmp_path/'bad.json',seed,['a']*4,events[:-1])


def test_replay_selection_tie_breaks_stably():
    rows=[dict(game_id=i,candidate_seat=0,vp=[vp,10,0,0],ranking=rank) for i,vp,rank in
          [(4,20,[0,1,2,3]),(1,20,[0,1,2,3]),(2,1,[3,0,1,2]),(3,9,[1,0,2,3]),(5,8,[3,0,1,2])]]
    chosen=select_replays(rows)
    assert chosen == dict(best_vp=1,lowest_vp=2,worst_rank=2,median_vp=3)
    assert select_replays(list(reversed(rows))) == chosen


def test_training_replay_cap_across_updates_and_labels(tmp_path):
    from types import SimpleNamespace
    from brass_rl.league import Lineup, LineupEntry
    from brass_rl.seeding import SeedStreams
    from brass_rl.config import EvalConfig
    streams=SeedStreams(0xD274A152EC66)
    lineup=Lineup((LineupEntry('current',trainable=True),LineupEntry('random'),LineupEntry('random'),LineupEntry('random')))
    assert EvalConfig().training_replay_limit == 10
    for game_id in range(3):
        seed=streams.seed('train.deal',game_id)
        agents=[brass.RandomBot(i+game_id) for i in range(4)]
        referee=RefereeGame(seed,agents)
        events=[]
        while not referee.is_over():
            seat,obs=referee.pending()
            events.append(referee.apply_index(agents[seat].act(obs)))
        ep=SimpleNamespace(game_id=game_id,lineup=lineup,events=events,result=referee.result())
        keep=retain_training_replays(tmp_path,[ep],streams,2,'policy-current',{})
        assert len(keep) <= 2
        assert len(list(tmp_path.glob('[0-9]*.json'))) <= 2
        if game_id in keep:
            assert json.loads((tmp_path/('%s.json' % game_id)).read_text())['seats'][0]['name'] == 'policy-current'
    assert len(json.loads((tmp_path/'index.json').read_text())) == 2
    retain_training_replays(tmp_path,[],streams,0,'policy-current',{})
    assert not list(tmp_path.glob('[0-9]*.json'))


def test_median_vp_ties_choose_lowest_game_id():
    rows=[dict(game_id=i,candidate_seat=0,vp=[10,5,2,0],ranking=[0,1,2,3]) for i in (9,2,7,5)]
    assert select_replays(rows)['median_vp'] == 2


def test_frozen_replay_identity_distinguishes_reused_local_ids(tmp_path):
    from types import SimpleNamespace
    from brass_rl.league import Lineup, LineupEntry
    from brass_rl.seeding import SeedStreams
    streams=SeedStreams(0x315E8639A)
    lineup=Lineup((LineupEntry('current',trainable=True),LineupEntry('frozen','initial-000000'),
                   LineupEntry('greedy'),LineupEntry('random')))
    episodes=[]
    for game_id in range(2):
        agents=[brass.RandomBot(i+game_id) for i in range(4)]
        referee=RefereeGame(streams.seed('train.deal',game_id),agents)
        events=[]
        while not referee.is_over():
            seat,obs=referee.pending()
            events.append(referee.apply_index(agents[seat].act(obs)))
        episodes.append(SimpleNamespace(game_id=game_id,lineup=lineup,events=events,result=referee.result()))
    frozen_labels=[]
    for run,digest in [('first','a'*64),('second','b'*64)]:
        directory=tmp_path/run
        identities={'initial-000000':'initial-000000:weights-sha256:'+digest}
        for episode in episodes:
            retain_training_replays(directory,[episode],streams,1,'policy-current',identities)
            files=list(directory.glob('[0-9]*.json'))
            assert len(files) == 1
            names=[seat['name'] for seat in json.loads(files[0].read_text())['seats']]
            assert names == ['policy-current','initial-000000:weights-sha256:'+digest,'greedy','random']
        frozen_labels.append(names[1])
    assert frozen_labels[0] != frozen_labels[1]
