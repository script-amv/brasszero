# SPDX-License-Identifier: Apache-2.0
"""Review regressions: no new complete games; terminal history is a retained fixture."""
import json
from pathlib import Path
import pytest
from brass_ai.replay_archive import Archive, Recorder, digest
from brass_search import SearchConfig
from test_evaluate import policy  # noqa: F401
from test_search_evaluate import local_inventory, controller  # noqa: F401


def test_reservation_precedes_full_policy_inspection(policy, tmp_path, local_inventory, monkeypatch):
    m=controller(); out=tmp_path/'run'
    def inspect(path):
        assert (out/'budget.json').exists(), 'weight inspection before durable reservation'
        budget=json.loads((out/'budget.json').read_text())
        assert budget['remaining_seconds']==0 and budget['reservation']['seconds']==1800
        raise RuntimeError('deliberate inspection failure')
    monkeypatch.setattr(m, 'inspect_policy', inspect)
    with pytest.raises(RuntimeError, match='deliberate inspection failure'):
        m.evaluate_search(policy, SearchConfig(), out, master_seed=123, stop_requested=lambda:True)


def test_loaded_object_must_match_pinned_identity(policy, tmp_path, local_inventory, monkeypatch):
    m=controller(); out=tmp_path/'run'
    from brass_rl.artifacts import export_model_policy
    from brass_rl.policy_io import load_policy
    model, _, _=load_policy(policy)
    other=load_policy(export_model_policy(model, 1, tmp_path/'other'))
    monkeypatch.setattr(m,'load_policy',lambda *a,**k:other)
    monkeypatch.setattr(m,'_play',lambda *a,**k:(_ for _ in ()).throw(AssertionError('mismatched weights reached gameplay')))
    with pytest.raises(ValueError,match='loaded policy'):
        m.evaluate_search(policy, SearchConfig(), out, master_seed=123)
    assert Archive().records()==[]


def test_unknown_cost_totals_are_null_and_coverage_explicit(policy,tmp_path,local_inventory):
    m=controller();suite=m.prepare_suite(policy,SearchConfig(),master_seed=8)
    rows=[dict(m._lineage(suite,g),ranking=[0,1,2,3],vp=[10,9,8,7],plies=200,telemetry=None,recovered=True) for g in suite['games']]
    report=m.build_report(tmp_path/'report',suite,rows,Archive(),'complete')
    costs=report['costs']
    for key in ('whole_agent_inference_calls','real_memory_update_calls','native_simulated_applies','completed_rounds','fallbacks','search_decisions','games_per_controller_hour'):
        assert costs[key] is None, key
    assert costs['measured_decisions']==0 and costs['missing_telemetry_games']==8
    assert costs['measured_work']['whole_agent_inference_calls']==0


def terminal_journal(policy,out):
    m=controller();m.evaluate_search(policy,SearchConfig(),out,master_seed=4,stop_requested=lambda:True)
    suite=json.loads((out/'suite.json').read_text())
    fixture=json.loads((Path(__file__).resolve().parents[2]/'engine/tools/tests/fixtures/strategy/record-0.json').read_text())
    suite['deal_seeds']=[fixture['initial']['seed']]
    for g in suite['games']:g['deal_seed']=fixture['initial']['seed']
    suite.pop('suite_id');suite['suite_id']=digest(suite)
    (out/'suite.json').write_text(json.dumps(suite))
    spec=suite['games'][0]
    recorder=Recorder(Archive(),seed=spec['deal_seed'],seats=m._seats(spec),provenance=m._provenance(suite,spec),game_id=m.replay_id(suite,spec))
    for move in fixture['moves']:recorder.append(move['player'],move['move'])
    recorder.interrupt()
    return fixture,recorder.game_id


def test_terminal_journal_recovers_with_no_allowance_or_policy(policy,tmp_path,local_inventory,monkeypatch):
    m=controller();out=tmp_path/'run';fixture,identity=terminal_journal(policy,out)
    from brass_rl.search_budget import WallBudget
    WallBudget(out,1800).reserve()
    from brass_rl import policy_io
    def forbidden(*a,**k):raise AssertionError('inspection constructed a policy')
    monkeypatch.setattr(policy_io,'load_policy',forbidden)
    report=m.evaluate_search(tmp_path/'missing-policy',None,out,master_seed=4,report_only=True)
    assert report['completed_games']==1
    assert report['results'][0]['vp']==fixture['result']['vp']
    assert report['results'][0]['telemetry'] is None
    assert Archive().read(identity)['status']=='completed'
    assert json.loads((out/'budget.json').read_text())['remaining_seconds']==0


def test_report_only_retains_stop_reason_and_ignores_gameplay_source_drift(policy,tmp_path,local_inventory,monkeypatch):
    m=controller();out=tmp_path/'run'
    m.evaluate_search(policy,SearchConfig(),out,master_seed=4,stop_requested=lambda:True)
    monkeypatch.setattr(m,'controller_source',lambda:{'new':'source'})
    report=m.evaluate_search(tmp_path/'missing',None,out,master_seed=4,report_only=True)
    assert report['stop_reason']=='requested'
    assert report['inspection']['mode']=='report_only'
    with pytest.raises(ValueError,match='immutable'):
        m.evaluate_search(policy,SearchConfig(),out,master_seed=4,stop_requested=lambda:True)


def test_nested_match_context_preserves_manifests_without_changing_ordinary_match(monkeypatch):
    import brass_ai.arena as arena
    from brass_ai.replay_archive import recording_context,recording_metadata
    seen=[]
    monkeypatch.setattr(arena,'play_game',lambda *a,**k:seen.append(recording_metadata()))
    arena.run_match(['random']*4,1,6)
    ordinary=list(seen);seen.clear()
    import hashlib
    experiment=hashlib.sha256(b'{"master_seed":6,"specs":["random","random","random","random"]}').hexdigest()
    assert ordinary == [dict(category='evaluation', experiment=experiment, game=r, deal=0,
                             rotation=r, interface='arena') for r in range(4)]
    manifest={'searched':{'id':'search:test','base_policy_id':'policy:test','base_update':600,'config':{'max_rollouts':12}}}
    with recording_context(search_players=manifest):
        arena.run_match(['random']*4,1,6)
    assert len(seen)==4
    for old,new in zip(ordinary,seen):
        assert new.pop('search_players',None)==manifest
        assert new==old
    assert recording_metadata()=={}


def test_validation_recording_category(policy,tmp_path,local_inventory):
    m=controller();checks=[0]
    def stop():checks[0]+=1;return checks[0]>4
    m.evaluate_search(policy,SearchConfig(max_rollouts=0),tmp_path/'run',master_seed=7,stop_requested=stop,category='validation')
    assert Archive().records()[0]['category']=='validation'


def test_exhausted_resume_preserves_unclean_reservation_and_unknown_work(policy,tmp_path,local_inventory):
    m=controller();out=tmp_path/'run'
    m.evaluate_search(policy,SearchConfig(),out,master_seed=4,stop_requested=lambda:True)
    from brass_rl.search_budget import WallBudget
    WallBudget(out,1800).reserve()
    original=(out/'budget.json').read_bytes()
    report=m.evaluate_search(policy,SearchConfig(),out,master_seed=4)
    assert (out/'budget.json').read_bytes()==original
    assert report['costs']['unclean_attempt_work_unknown']
    assert report['costs']['whole_agent_inference_calls'] is None


@pytest.mark.parametrize('location', ['direct','descendant','symlink'])
def test_published_policy_output_guard_preserves_all_bytes(policy,tmp_path,local_inventory,location):
    import hashlib
    def snapshot():
        return {str(p.relative_to(policy)):hashlib.sha256(p.read_bytes()).hexdigest() for p in policy.rglob('*') if p.is_file()}
    original=snapshot()
    if location=='direct':out=policy
    elif location=='descendant':out=policy/'child'/'run'
    else:
        out=tmp_path/'alias';out.symlink_to(policy,target_is_directory=True)
    with pytest.raises(ValueError,match='published|checkpoint|immutable'):
        controller().evaluate_search(policy,SearchConfig(),out,master_seed=1,stop_requested=lambda:True)
    assert snapshot()==original
    assert not (policy/'child').exists()


def test_failed_retry_after_measured_partial_keeps_unknown_work(policy,tmp_path,local_inventory,monkeypatch):
    m=controller();out=tmp_path/'run';checks=[0];config=SearchConfig(max_rollouts=0)
    def stop():checks[0]+=1;return checks[0]>6
    first=m.evaluate_search(policy,config,out,master_seed=42,stop_requested=stop)
    known=first['costs']['measured_work']['whole_agent_inference_calls']
    original_plain,original_search=m.TorchAgent.act,m.SearchAgent.act
    def plain(self,obs):original_plain(self,obs);raise RuntimeError('failed retry after forward')
    def search(self,obs):original_search(self,obs);raise RuntimeError('failed retry after forward')
    monkeypatch.setattr(m.TorchAgent,'act',plain);monkeypatch.setattr(m.SearchAgent,'act',search)
    with pytest.raises(RuntimeError,match='failed retry'):
        m.evaluate_search(policy,config,out,master_seed=42)
    for report in (json.loads((out/'report.json').read_text()),
                   m.evaluate_search(None,None,out,master_seed=42,report_only=True)):
        assert not report['costs']['whole_run_work_complete']
        assert report['costs']['whole_agent_inference_calls'] is None
        assert report['costs']['measured_work']['whole_agent_inference_calls']==known
        assert report['costs']['unmeasured_attempts']==1


def test_inspection_rejects_different_native_binary_before_terminal_publication(policy,tmp_path,local_inventory,monkeypatch):
    import brass
    m=controller();out=tmp_path/'run';_,identity=terminal_journal(policy,out)
    archive=Archive();journal=archive.path(identity,'.jsonl');before=journal.read_bytes()
    other=tmp_path/'other-native.so';other.write_bytes(b'not the pinned extension bytes')
    monkeypatch.setattr(brass.brass,'__file__',str(other))
    with pytest.raises(ValueError,match='native.*binary|extension'):
        m.evaluate_search(None,None,out,master_seed=4,report_only=True)
    assert journal.read_bytes()==before and not archive.path(identity).exists()
    assert not list((out/'games').glob('*.json'))


@pytest.mark.parametrize('published', [False, True], ids=['terminal-journal', 'completed-record'])
def test_recovery_rejects_embedded_game_id_before_publication(policy, tmp_path, local_inventory, published):
    import gzip
    m = controller()
    out = tmp_path/'run'
    fixture, identity = terminal_journal(policy, out)
    suite = json.loads((out/'suite.json').read_text())
    wrong_id = m.replay_id(suite, suite['games'][1])
    archive = Archive()
    journal = archive.path(identity, '.jsonl')
    if published:
        record = dict(archive.read(identity), status='completed', result=fixture['result'], game_id=wrong_id)
        archive.path(identity).write_bytes(gzip.compress(json.dumps(record).encode()))
        journal.unlink()
    else:
        header, moves = journal.read_bytes().split(b'\n', 1)
        journal.write_bytes(json.dumps(dict(json.loads(header), game_id=wrong_id)).encode() + b'\n' + moves)
    before = {p: p.read_bytes() for p in archive.games.iterdir()}
    assert not archive.path(wrong_id).exists()
    with pytest.raises(ValueError, match='archive.*(manifest|game ID)'):
        m.evaluate_search(None, None, out, master_seed=4, report_only=True)
    assert {p: p.read_bytes() for p in archive.games.iterdir()} == before
    assert not list((out/'games').glob('*.json'))


def test_first_checkpoint_typo_reuses_original_allowance(policy, tmp_path, local_inventory, monkeypatch):
    from brass_rl import __main__ as cli
    m = controller()
    out = tmp_path/'run'
    config = tmp_path/'config.json'
    config.write_text('{}')
    arguments = ['--config', str(config), '--out', str(out), '--master-seed', '4']
    monkeypatch.setattr(cli._SearchStop, '__call__', lambda self: True)
    with pytest.raises(SystemExit) as error:
        cli.main(['search-eval', '--checkpoint', str(tmp_path/'typo'), *arguments])
    assert error.value.code == 2
    assert {p.name for p in out.iterdir()} == {'.writer.lock', 'budget.json'}
    first = json.loads((out/'budget.json').read_text())
    assert first['reservation'] is None and 0 < first['remaining_seconds'] < 1800

    inspect = m.inspect_policy
    def inspect_retry(path):
        reserved = json.loads((out/'budget.json').read_text())
        assert reserved['reservation']['seconds'] == first['remaining_seconds']
        assert reserved['measured_seconds'] == first['measured_seconds']
        return inspect(path)
    monkeypatch.setattr(m, 'inspect_policy', inspect_retry)
    cli.main(['search-eval', '--checkpoint', str(policy), *arguments])
    final = json.loads((out/'budget.json').read_text())
    report = json.loads((out/'report.json').read_text())
    assert final['reservation'] is None and 0 < final['remaining_seconds'] < first['remaining_seconds']
    assert report['stop_reason'] == 'requested' and report['completed_games'] == 0
    assert Archive().records() == []


@pytest.mark.parametrize('condition', [
    'crash', 'unrelated-file', 'wrong-format', 'empty-object', 'inconsistent-balance', 'exhausted', 'wrong-limit',
])
def test_pre_suite_retry_refuses_unusable_allowance(tmp_path, condition):
    from brass_rl.search_budget import RunInvocation, WallBudget
    budget = WallBudget(tmp_path, 10)
    budget.reserve(now=0)
    if condition != 'crash':
        budget.finish(now=10 if condition == 'exhausted' else 2)
    if condition == 'unrelated-file':
        (tmp_path/'other').write_text('keep')
    if condition == 'wrong-format':
        budget.path.write_text(json.dumps(dict(budget.state, format='other')))
    if condition == 'empty-object':
        budget.path.write_text('{}')
    if condition == 'inconsistent-balance':
        budget.path.write_text(json.dumps(dict(budget.state, remaining_seconds=10)))
    protected = {p: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        with RunInvocation(tmp_path, 11 if condition == 'wrong-limit' else 10):
            pytest.fail('pre-suite retry unexpectedly acquired the allowance')
    assert all(path.read_bytes() == raw for path, raw in protected.items())
    assert not (tmp_path/'suite.json').exists()


def test_missing_checkpoint_input_preserves_existing_rows_and_allows_retry(policy,tmp_path,local_inventory):
    m=controller();out=tmp_path/'run';_,identity=terminal_journal(policy,out)
    first=m.evaluate_search(None,None,out,master_seed=4,report_only=True)
    assert first['completed_games']==1
    paths=[out/'report.json']+list((out/'games').glob('*.json'))
    before={p:p.read_bytes() for p in paths}
    with pytest.raises((OSError,ValueError)):
        m.evaluate_search(tmp_path/'typo',SearchConfig(),out,master_seed=4,stop_requested=lambda:True)
    assert not (out/'failure.json').exists()
    assert all(p.read_bytes()==raw for p,raw in before.items())
    retried=m.evaluate_search(policy,SearchConfig(),out,master_seed=4,stop_requested=lambda:True)
    assert retried['completed_games']==1 and retried['stop_reason']=='requested'


@pytest.mark.parametrize('result_field',[None,'missing'])
def test_completed_archive_requires_stored_native_outcome(result_field):
    m=controller()
    fixture=json.loads((Path(__file__).resolve().parents[2]/'engine/tools/tests/fixtures/strategy/record-0.json').read_text())
    fixture['status']='completed'
    if result_field=='missing':fixture.pop('result')
    else:fixture['result']=None
    with pytest.raises(ValueError,match='outcome'):
        m.recover_record(fixture)


def retained_measured_row(policy,out):
    """Retained native completion plus explicit cost fixture; no agent decisions."""
    m=controller();fixture,identity=terminal_journal(policy,out)
    m.evaluate_search(None,None,out,master_seed=4,report_only=True)
    row_path=out/'games'/'0000.json';row=json.loads(row_path.read_text())
    telemetry=dict(seconds=2.,decision_latencies=[{'seat':a['player'],'seconds':.01} for a in fixture['moves']],
                   search_decisions=[],real_memory_update_calls=len(fixture['moves'])*7,
                   plain_policy_forward_calls=len(fixture['moves']),reconstructed_decisions=0,
                   unapplied_decisions=0,peak_rss_bytes=1024,
                   whole_agent_inference_calls=len(fixture['moves'])*8)
    row.update(telemetry=telemetry,recovered=False,attempt_id='completed-attempt')
    row_path.write_text(json.dumps(row))
    return fixture,row,telemetry


@pytest.mark.parametrize('older_unknown,unclean',[(False,False),(True,False),(False,True)])
def test_completed_row_reconciles_only_its_pending_marker(policy,tmp_path,local_inventory,older_unknown,unclean):
    m=controller();out=tmp_path/'run';fixture,row,telemetry=retained_measured_row(policy,out)
    attempt_dir=out/'attempts';attempt_dir.mkdir()
    common=dict(suite_id=row['suite_id'],game=row['game'],partial=True,telemetry=None)
    (attempt_dir/'000001.json').write_text(json.dumps(dict(common,attempt_id='completed-attempt')))
    if older_unknown:
        (attempt_dir/'000000.json').write_text(json.dumps(dict(common,attempt_id='earlier-attempt')))
    if unclean:
        from brass_rl.search_budget import WallBudget
        WallBudget(out,1800).reserve()
    budget_before=(out/'budget.json').read_bytes()
    report=m.evaluate_search(None,None,out,master_seed=4,report_only=True)
    costs=report['costs']
    assert costs['measured_attempts']==1
    assert costs['measured_decisions']==len(fixture['moves'])
    assert costs['unmeasured_attempts']==int(older_unknown)
    assert costs['partial_attempts']==int(older_unknown)
    assert costs['measured_work']['whole_agent_inference_calls']==telemetry['whole_agent_inference_calls']
    assert costs['unclean_attempt_work_unknown']==unclean
    assert costs['whole_run_work_complete']==(not older_unknown and not unclean)
    if unclean:assert costs['games_per_controller_hour'] is None
    assert (out/'budget.json').read_bytes()==budget_before


def test_completed_row_keeps_attempt_id_when_marker_unlink_fails(policy,tmp_path,local_inventory,monkeypatch):
    import brass
    m=controller();out=tmp_path/'run';fixture,prior,telemetry=retained_measured_row(policy,out)
    # Reconstruct saved choices into the controller's next recording. No policy
    # chooses anything and no fresh history/game is generated.
    def retained_play(loaded,config,suite,spec,archive,stop):
        native=brass.Game(fixture['initial']['seed'])
        recorder=Recorder(archive,seed=spec['deal_seed'],seats=m._seats(spec),
                          provenance=m._provenance(suite,spec),game_id=m.replay_id(suite,spec))
        for action in fixture['moves']:
            recorder.append(action['player'],action['move']);native.apply(json.dumps(action['move']))
        recorder.finish(native)
        return dict(vp=fixture['result']['vp'],ranking=fixture['result']['ranking'],plies=len(fixture['moves']),
                    telemetry=telemetry,recovered=False)
    monkeypatch.setattr(m,'_play',retained_play)
    unlink=Path.unlink
    def fail_marker(path,*args,**kwargs):
        if path.parent==out/'attempts' and path.name=='000000.json':
            raise OSError('marker unlink failed')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',fail_marker)
    with pytest.raises(OSError,match='marker unlink failed'):
        m.evaluate_search(policy,SearchConfig(),out,master_seed=4)
    saved=json.loads((out/'games'/'0001.json').read_text())
    pending=json.loads((out/'attempts'/'000000.json').read_text())
    assert saved['attempt_id']==pending['attempt_id']
    for report in (json.loads((out/'report.json').read_text()),m.evaluate_search(None,None,out,master_seed=4,report_only=True)):
        assert report['costs']['measured_attempts']==2
        assert report['costs']['unmeasured_attempts']==0
        assert report['costs']['whole_run_work_complete']
        assert report['costs']['whole_agent_inference_calls']==2*telemetry['whole_agent_inference_calls']
