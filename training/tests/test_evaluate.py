# SPDX-License-Identifier: Apache-2.0
"""Unseen, resumable four-rotation evaluation and deal-cluster statistics."""
from dataclasses import replace
import json
import pytest
from brass_rl.evaluate import EvalSuite, cluster_summary, evaluate, choose_development_best
from brass_rl.seeding import SeedStreams


def fixture_rows():
    return [dict(game_id=d*4+r, deal_id=d, rotation=r, lineup='greedy',
                 labels=['candidate','greedy','greedy','greedy'], candidate_seat=0,
                 ranking=[1,0,2,3] if d == 0 else [0,1,2,3],
                 vp=[0,10,2,1] if d == 0 else [10,0,2,1])
            for d in range(2) for r in range(4)]


def test_rotations_form_one_deal_cluster():
    rows = fixture_rows()
    summary = cluster_summary(rows, 'candidate', 123)
    assert summary.win_rate == .5
    assert summary.independent_deals == 2 and summary.seat_games == 8
    assert summary.win_ci == (0.,1.)
    assert summary == cluster_summary(rows, 'candidate', 123)
    assert -10 <= summary.margin_ci[0] <= summary.margin_ci[1] <= 8
    with pytest.raises(ValueError, match='rotation'):
        cluster_summary(rows[:-1], 'candidate', 123)


def test_ties_and_lineup_labels():
    rows = fixture_rows()[:4]
    for row in rows:
        row['vp'] = [10,10,0,0]
        row['ranking'] = [0,0,2,2]
    summary = cluster_summary(rows, 'candidate', 42)
    assert summary.win_rate == 0 and summary.tied_first == 4
    assert summary.mean_rank == 1 and summary.vp_margin == 0


def test_eval_streams_disjoint_and_manifests_reused(tmp_path, monkeypatch, policy):
    import brass_rl.evaluate as module
    from brass_rl.policy_io import inspect_policy
    identity = inspect_policy(policy).id
    suite = EvalSuite('dev', 914235, 2, {'greedy':['candidate','greedy','greedy','greedy']}, identity)
    calls = []
    def game(candidate, specs, seed, sampling):
        calls.append(seed)
        return dict(ranking=[0,1,2,3], vp=[10,5,3,0], plies=0, events=())
    monkeypatch.setattr(module, '_play_game', game)
    report = evaluate(policy, suite, tmp_path/'eval')
    assert len(report.results) == 8 and len(calls) == 8
    assert len(set(calls)) == 2
    assert len({SeedStreams(suite.master_seed).seed(s+'.deal', d) for s in ('train','dev','final') for d in range(2)}) == 6
    assert evaluate(policy, suite, tmp_path/'eval').results == report.results
    assert len(calls) == 8
    with pytest.raises(ValueError, match='immutable'):
        evaluate(policy, replace(suite,deals=3), tmp_path/'eval')


def test_final_suite_requires_preselected_candidate(tmp_path, policy, monkeypatch):
    import brass_rl.evaluate as module
    from brass_rl.policy_io import inspect_policy
    with pytest.raises(ValueError, match='candidate'):
        EvalSuite('final', 123, 100, {'greedy':['candidate','greedy','greedy','greedy']}, '')
    suite=EvalSuite('final',123,100,{'greedy':['candidate','greedy','greedy','greedy']},inspect_policy(policy).id)
    def interrupt(*a):
        raise RuntimeError('interrupted')
    monkeypatch.setattr(module,'_play_game',interrupt)
    with pytest.raises(RuntimeError):
        evaluate(policy,suite,tmp_path/'eval')
    manifest=json.loads((tmp_path/'eval'/'suite.json').read_text())
    assert manifest['candidate_id'] == suite.candidate_id and manifest['deals'] == 100
    with pytest.raises(ValueError,match='candidate'):
        evaluate(policy,replace(suite,candidate_id='other'),tmp_path/'eval')
    from brass_rl.artifacts import export_model_policy
    from brass_rl.policy_io import load_policy
    other=export_model_policy(load_policy(policy)[0],1,tmp_path/'other-policy')
    with pytest.raises(ValueError,match='immutable'):
        evaluate(other,replace(suite,candidate_id=inspect_policy(other).id),tmp_path/'eval')


def test_selection_excludes_baselines_versions_and_breaks_ties():
    rows=[dict(candidate_id='old',suite_id='v1',win_rate=1.,mean_rank=1.,update=1),
          dict(candidate_id='early',suite_id='v2',win_rate=.5,mean_rank=2.,update=2)]
    current=dict(candidate_id='now',suite_id='v2',win_rate=.5,mean_rank=2.,update=3)
    assert choose_development_best(rows,current) == 'early'
    assert choose_development_best(rows,dict(current,suite_id='v3')) == 'now'


@pytest.fixture
def policy(tmp_path):
    from brass_rl.artifacts import export_model_policy
    from brass_rl.model import BrassPolicy
    from brass_rl.config import ModelConfig
    from brass_rl.schema import ComponentData
    components=ComponentData.load()
    model=BrassPolicy(ModelConfig(entity_dim=4,context_dim=8,recurrent_dim=4,message_passing_rounds=1),components.schema,components)
    return export_model_policy(model,0,tmp_path/'policy')


def test_evaluation_preserves_global_rng_on_failure(tmp_path, policy, monkeypatch):
    import random
    import numpy as np
    import torch
    import brass_rl.evaluate as module
    from brass_rl.artifacts import capture_rng_states
    from brass_rl.policy_io import inspect_policy
    from test_train import assert_nested_equal
    suite=EvalSuite('dev',123,1,{'greedy':['candidate','greedy','greedy','greedy']},inspect_policy(policy).id)
    before=capture_rng_states()
    def fail(*a):
        random.random(); np.random.random(); torch.rand(1)
        raise RuntimeError('evaluation failure')
    monkeypatch.setattr(module,'_play_game',fail)
    with pytest.raises(RuntimeError,match='evaluation failure'):
        evaluate(policy,suite,tmp_path/'eval')
    assert_nested_equal(before,capture_rng_states())


@pytest.mark.parametrize('field,value',[('deals',0),('deals',True),('replay_limit',101),('training_replay_limit',-1)])
def test_evaluation_bounds_validated(field,value):
    from brass_rl.config import EvalConfig
    with pytest.raises(ValueError,match=field):
        EvalConfig(**{field:value})


def test_candidate_identity_can_equal_new_holdout():
    rows=fixture_rows()
    for row in rows:
        row['labels']=['candidate']*4
    summary=cluster_summary(rows,'candidate',42)
    assert summary.seat_games == 8 and summary.win_rate == .5


def test_interrupted_rows_reused_without_duplicates(tmp_path,policy,monkeypatch):
    import brass_rl.evaluate as module
    from brass_rl.policy_io import inspect_policy
    suite=EvalSuite('dev',578,2,{'greedy':['candidate','greedy','greedy','greedy']},inspect_policy(policy).id,0)
    calls=[]
    def game(*args):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError('fourth game interruption')
        return dict(ranking=[0,1,2,3],vp=[10,5,2,0],plies=0,events=())
    monkeypatch.setattr(module,'_play_game',game)
    with pytest.raises(RuntimeError):
        evaluate(policy,suite,tmp_path/'eval')
    assert len(list((tmp_path/'eval').rglob('games/*.json'))) == 3
    report=evaluate(policy,suite,tmp_path/'eval')
    assert len(report.results) == 8 and len(calls) == 9
    assert len(list((tmp_path/'eval').rglob('games/*.json'))) == 8


def test_missing_historical_exemplars_are_never_regenerated(tmp_path,policy,monkeypatch):
    import brass_rl.evaluate as module
    from brass_rl.policy_io import inspect_policy
    suite=EvalSuite('dev',578,2,{'greedy':['candidate','greedy','greedy','greedy']},inspect_policy(policy).id,4)
    calls=[]
    scores=[10,20,30,40,100,101,102,103]
    def game(candidate,specs,seed,sampling):
        # Deal/rotation seeds make regeneration deterministic without a real neural game.
        key=tuple(sampling)
        if key not in calls:
            calls.append(key)
        i=calls.index(key)
        seat=specs.index('candidate')
        vp=[50]*4; vp[seat]=scores[i]
        return dict(ranking=[0,1,2,3],vp=vp,plies=1,events=('recorded',))
    def replay(path,*args,**kwargs):
        path.write_text('{}')
    monkeypatch.setattr(module,'_play_game',game)
    monkeypatch.setattr(module,'write_gui_replay',replay)
    report=evaluate(policy,suite,tmp_path/'eval')
    expected={0,1,7}  # Discarded historical median stays missing; no regeneration.
    files=list((tmp_path/'eval').rglob('replays/*.json'))
    assert {int(p.stem) for p in files} == expected
    assert len(files) <= 4 and len(calls) == 8
    assert evaluate(policy,suite,tmp_path/'eval').results == report.results
    assert {int(p.stem) for p in (tmp_path/'eval').rglob('replays/*.json')} == expected


@pytest.mark.parametrize('changed',[
    'players/baselines/src/greedy.rs','engine/bindings/src/lib.rs','engine/tools/brass_ai/arena.py',
    'training/brass_rl/evaluate.py','engine/core/src/lib.rs'])
def test_source_changes_version_baseline_suites(tmp_path,monkeypatch,changed):
    import shutil
    import brass_rl.evaluate as module
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    for name in ('players/baselines','engine/bindings/src','engine/core'):
        shutil.copytree(root/name,tmp_path/name,ignore=shutil.ignore_patterns('target','tests','benches','PERF.md'))
    for name in ('Cargo.lock','engine/bindings/Cargo.toml','engine/tools/brass_ai/arena.py','engine/tools/brass_ai/events.py',
                 'engine/tools/brass_ai/seeds.py','engine/tools/brass_ai/agents.py','engine/tools/brass_ai/fingerprints.py',
                 'training/brass_rl/evaluate.py'):
        dest=tmp_path/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest)
    before=module._opponent_provenance(tmp_path)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    suite=EvalSuite('dev',123,1,{'mixed':['candidate','random','greedy','greedy']},'candidate-id')
    manifest=suite.manifest()
    path=tmp_path/changed
    path.write_bytes(path.read_bytes()+b'\n// provenance change\n')
    assert module._opponent_provenance(tmp_path) != before
    assert suite.manifest()['suite_id'] != manifest['suite_id']


def test_missing_opponent_sources_fail_clearly(tmp_path):
    import brass_rl.evaluate as module
    with pytest.raises(ValueError,match='source'):
        module._opponent_provenance(tmp_path)


def test_baselines_use_native_constructors_not_mutable_registry(monkeypatch):
    import brass_ai.agents as agents
    import brass_rl.evaluate as module
    def poisoned(seed):
        raise RuntimeError('mutable registry substituted opponent')
    monkeypatch.setitem(agents._REGISTRY,'random',poisoned)
    monkeypatch.setitem(agents._REGISTRY,'greedy',poisoned)
    result=module._play_game(None,['random','greedy','random','greedy'],123,[1,2,3,4])
    assert result['plies'] > 100 and len(result['events']) == result['plies']
    assert len(result['vp']) == len(result['ranking']) == 4


def test_actual_evaluation_archives_all_games_with_zero_exemplars(tmp_path, policy):
    import brass
    from brass_ai.replay_archive import Archive
    from brass_rl.policy_io import inspect_policy
    identity = inspect_policy(policy).id
    suite = EvalSuite('dev', 9364, 1, {'random':['candidate','random','random','random']}, identity, 0)
    report = evaluate(policy, suite, tmp_path/'eval')
    archive = Archive()
    records = {r['provenance']['game']:r for r in archive.records()}
    assert len(records) == 4
    assert not list((tmp_path/'eval').rglob('replays/*.json'))
    for row in report.results:
        record = records[row['game_id']]
        game = brass.Game(record['initial']['seed'])
        for action in record['moves']:
            assert action['player'] == game.current_player()
            game.apply(json.dumps(action['move']))
        assert game.is_over()
        assert list(game.vp()) == row['vp'] == record['result']['vp']
        assert list(game.ranking()) == row['ranking'] == record['result']['ranking']
        assert record['seats'][row['candidate_seat']]['bot'] == identity
        samples = row['value_predictions']
        assert [s['ply'] for s in samples] == [i for i, move in enumerate(record['moves'])
                                               if move['player'] == row['candidate_seat']]
        assert all(len(s['values']) == 4 for s in samples)
        assert row['terminal_brier']['decisions'] == len(samples)
        assert 0 <= row['terminal_brier']['brier'] <= 1
        assert len(row['first_view_sha256']) == 64
        assert 'value_predictions' not in record and all('values' not in move for move in record['moves'])
    assert evaluate(policy, suite, tmp_path/'eval').results == report.results
    assert len(archive.rebuild()) == 4


def test_outcome_only_eval_never_scores_value_head(tmp_path, monkeypatch):
    from brass_rl import evaluate as module
    from brass_rl.artifacts import export_model_policy
    from brass_rl.model import BrassPolicy
    from brass_rl.schema import ComponentData
    from brass_rl.config import ModelConfig
    from brass_rl.policy_io import load_policy
    components = ComponentData.load()
    policy = export_model_policy(BrassPolicy(ModelConfig(), components.schema, components), 2, tmp_path/'frozen')
    loaded = load_policy(policy)
    monkeypatch.setattr(module, 'terminal_brier', lambda *args: pytest.fail('score head was labeled a win forecast'))
    from brass_ai.replay_archive import Archive, recording_context
    with recording_context(outcomes_only=True, category='qa-four-branch'):
        game = module._play_game(loaded, ['candidate', 'greedy', 'greedy', 'greedy'], 2**64-1, [1, 2, 3, 4])
    assert 'value_predictions' not in game and 'terminal_brier' not in game
    replay = Archive().read(game['replay_id'])
    from brass_rl.transfer import validate_replay
    validate_replay(replay)
    assert replay['initial']['seed'] == 2**64-1
    assert replay['result']['vp'] == game['vp']
