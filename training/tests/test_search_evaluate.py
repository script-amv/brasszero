# SPDX-License-Identifier: Apache-2.0
"""Paired controller durability: real native prefixes, isolated archives, no pilot."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
from brass_search import SearchConfig
from test_evaluate import policy  # noqa: F401


def controller():
    from brass_rl import search_evaluate
    return search_evaluate


@pytest.fixture
def local_inventory(tmp_path, monkeypatch):
    module = controller()
    root = tmp_path/'inventory'
    root.mkdir()
    monkeypatch.setattr(module, 'INVENTORY_ROOT', root)
    from functools import partial
    monkeypatch.setattr(module, 'evaluate_search', partial(module.evaluate_search, category='validation'))
    return root


def test_paired_manifest_and_uint64(policy, tmp_path, local_inventory):
    m = controller()
    suite = m.prepare_suite(policy, SearchConfig(), master_seed=2**64-1)
    assert suite['format'] == 'brass-search-eval/1'
    assert suite['master_seed'] == 2**64-1
    assert len(suite['games']) == 8
    assert [g['candidate_seat'] for g in suite['games']] == [0,0,1,1,2,2,3,3]
    for search, control in zip(suite['games'][::2], suite['games'][1::2]):
        assert search['treatment'] == 'search' and control['treatment'] == 'control'
        assert search['sampling'] == control['sampling']
        assert len(set(search['sampling'])) == 4
        assert search['deal_seed'] == control['deal_seed']
        assert search['labels'].count(suite['search']['id']) == 1
        assert control['labels'] == [suite['baseline']['id']]*4
    assert suite['search']['id'] != suite['baseline']['id']
    assert suite['controller_source'] and suite['runtime']
    assert json.loads(json.dumps(suite)) == suite


def test_inventory_rejects_declared_deal(policy, local_inventory):
    m = controller()
    first = m.prepare_suite(policy, SearchConfig(), master_seed=10)
    (local_inventory/'prior.json').write_text(json.dumps({'deal_seeds': first['deal_seeds']}))
    second = m.prepare_suite(policy, SearchConfig(), master_seed=10)
    assert not set(first['deal_seeds']) & set(second['deal_seeds'])
    assert second['seed_inventory']['count'] == 1


def test_no_play_stop_and_immutable_resume(policy, tmp_path, local_inventory):
    m = controller(); out = tmp_path/'run'
    report = m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)
    assert report['completed_games'] == 0 and report['stop_reason'] == 'requested'
    assert report['remaining_games'] == 8
    for kwargs in ({'master_seed': 34}, {'master_seed': 33, 'wall_seconds': 30}, {'master_seed': 33, 'deals': 2}):
        with pytest.raises(ValueError, match='immutable'):
            m.evaluate_search(policy, SearchConfig(), out, **kwargs)
    with pytest.raises(ValueError, match='immutable'):
        m.evaluate_search(policy, replace(SearchConfig(), max_rollouts=0), out, master_seed=33)
    budget = (out/'budget.json').read_bytes()
    m.evaluate_search(policy, SearchConfig(), out, master_seed=33, report_only=True)
    assert (out/'budget.json').read_bytes() == budget
    with pytest.raises(ValueError, match='existing'):
        m.evaluate_search(policy, SearchConfig(), tmp_path/'missing', master_seed=33, report_only=True)


def test_runtime_or_controller_change_refused(policy, tmp_path, local_inventory, monkeypatch):
    m = controller(); out = tmp_path/'run'
    m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)
    monkeypatch.setattr(m, 'runtime_manifest', lambda: {'changed': True})
    with pytest.raises(ValueError, match='immutable'):
        m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)


def test_budget_reservation_crash_refund_and_exclusive_lock(tmp_path):
    from brass_rl.search_budget import RunLock, WallBudget
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError, match='writer'):
            with RunLock(tmp_path):
                pass
        budget = WallBudget(tmp_path, 10)
        budget.reserve(now=100)
        assert json.loads((tmp_path/'budget.json').read_text())['remaining_seconds'] == 0
        assert WallBudget(tmp_path, 10).remaining == 0  # no refund on an unclean exit
        budget.finish(now=103)
        assert WallBudget(tmp_path, 10).remaining == 7
        budget = WallBudget(tmp_path, 10)
        budget.reserve(now=200); budget.finish(now=220)
        assert WallBudget(tmp_path, 10).remaining == 0


def test_native_prefix_resume_does_not_duplicate(policy, tmp_path, local_inventory):
    from brass_ai.replay_archive import Archive
    m = controller(); out = tmp_path/'run'; config = SearchConfig(max_rollouts=0)
    checks = [0]
    def stop():
        checks[0] += 1
        return checks[0] > 6
    one = m.evaluate_search(policy, config, out, master_seed=999, stop_requested=stop)
    records = Archive().records()
    assert len(records) == 1 and 0 < len(records[0]['moves']) < 10
    before = records[0]['moves']
    checks[0] = 0
    two = m.evaluate_search(policy, config, out, master_seed=999, stop_requested=stop)
    after = Archive().records()
    assert len(after) == 1 and after[0]['moves'] == before
    assert one['completed_games'] == two['completed_games'] == 0
    assert after[0]['seats'][0]['bot'].startswith('search-sha256-v1:')
    assert after[0]['provenance']['search']['base_policy_id'].startswith('policy-sha256-v1:')


def test_storage_cap_preserves_records(policy, tmp_path, local_inventory):
    m = controller(); out = tmp_path/'run'
    report = m.evaluate_search(policy, SearchConfig(), out, master_seed=5, disk_limit_bytes=65536)
    assert report['stop_reason'] == 'disk_limit' and report['completed_games'] == 0
    assert (out/'suite.json').exists()


def test_completed_archive_recovery_never_loads_policy(policy, tmp_path, local_inventory, monkeypatch):
    import brass
    m = controller(); out = tmp_path/'run'; config = SearchConfig(max_rollouts=0)
    m.evaluate_search(policy, config, out, master_seed=123, stop_requested=lambda: True)
    # Reconstruct an existing complete history, not a newly played QA game.
    fixture = json.loads((Path(__file__).resolve().parents[2]/'engine/tools/tests/fixtures/strategy/record-0.json').read_text())
    # Recovery requires the declared seed; use a native terminal replay fixture in
    # the dedicated recovery helper, then reject wrong manifest at the controller.
    game = brass.Game(fixture['initial']['seed'])
    for move in fixture['moves']:
        game.apply(json.dumps(move['move']))
    assert game.is_over()
    row = m.recover_record(fixture)
    assert row['vp'] == list(game.vp()) and row['ranking'] == list(game.ranking())
    assert row['telemetry'] is None and row['recovered'] is True
    broken = dict(fixture, result=dict(fixture['result'], vp=[-1]*4))
    with pytest.raises(ValueError, match='outcome'):
        m.recover_record(broken)


def test_partial_attempt_costs_and_reconstruction_are_disclosed(policy, tmp_path, local_inventory):
    m = controller(); out = tmp_path/'run'; checks = [0]
    def stop():
        checks[0] += 1
        return checks[0] > 7
    report = m.evaluate_search(policy, SearchConfig(max_rollouts=0), out, master_seed=77, stop_requested=stop)
    assert report['costs']['partial_attempts'] == 1
    assert report['costs']['partial_decisions'] > 0
    assert report['costs']['whole_agent_inference_calls'] > 0
    assert len(report['partial_games']) == 1
    assert report['partial_games'][0]['plies'] == report['costs']['partial_decisions']-report['costs']['unapplied_decisions']


def test_completed_game_recovered_no_policy_replay_and_row_tamper_refused(policy, tmp_path, local_inventory, monkeypatch):
    from brass_ai.replay_archive import Archive
    m = controller(); out = tmp_path/'run'; config = SearchConfig(max_rollouts=0)
    def one_game():
        return len(list((out/'games').glob('*.json'))) >= 1
    report = m.evaluate_search(policy, config, out, master_seed=666, stop_requested=one_game)
    assert report['completed_games'] == 1 and report['remaining_games'] == 7
    assert report['results'][0]['telemetry']['real_memory_update_calls'] == report['results'][0]['plies']*7
    row_path = next((out/'games').glob('*.json'))
    expected = json.loads(row_path.read_text()); row_path.unlink()
    def fail(*args, **kwargs):
        raise AssertionError('recovery loaded policy')
    monkeypatch.setattr(m, 'load_policy', fail)
    recovered = m.evaluate_search(policy, config, out, master_seed=666, report_only=True)
    assert recovered['completed_games'] == 1
    assert recovered['results'][0]['telemetry'] is None
    assert recovered['results'][0]['vp'] == expected['vp']
    assert recovered['costs']['missing_telemetry_games'] == 1
    assert len(Archive().records()) == 1
    row = json.loads(row_path.read_text()); row['sampling'][0] += 1
    row_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='immutable completed row'):
        m.evaluate_search(policy, config, out, master_seed=666, report_only=True)


def test_missing_allowance_file_is_not_recreated(policy, tmp_path, local_inventory):
    m = controller(); out = tmp_path/'run'
    m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)
    (out/'budget.json').unlink()
    with pytest.raises(ValueError, match='allowance'):
        m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)


def test_archive_root_change_refused(policy, tmp_path, local_inventory, monkeypatch):
    m = controller(); out = tmp_path/'run'
    m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(tmp_path/'other-archive'))
    with pytest.raises(ValueError, match='archive'):
        m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)


def test_declared_training_and_development_ranges_excluded(local_inventory):
    m = controller()
    (local_inventory/'config.json').write_text(json.dumps({'master_seed': 1234, 'updates': 3,
        'games_per_update': 2, 'eval': {'deals': 2}}))
    from brass_rl.seeding import SeedStreams
    seeds, _ = m.declared_seeds(local_inventory)
    streams = SeedStreams(1234)
    assert streams.seed('train.deal', 5) in seeds
    assert streams.seed('dev.deal', 1) in seeds
    assert streams.seed('final.deal', 1) in seeds


def test_budget_exhaustion_resume_cannot_play(policy, tmp_path, local_inventory, monkeypatch):
    m = controller(); out = tmp_path/'run'
    m.evaluate_search(policy, SearchConfig(), out, master_seed=33, stop_requested=lambda: True)
    from brass_rl.search_budget import WallBudget
    WallBudget(out, 1800).reserve()  # crash after durable reservation, before clean refund
    def forbidden(*args, **kwargs):
        raise AssertionError('exhausted controller attempted model load')
    monkeypatch.setattr(m, 'load_policy', forbidden)
    report = m.evaluate_search(policy, SearchConfig(), out, master_seed=33)
    assert report['stop_reason'] == 'wall_limit'
    assert report['budget']['remaining_seconds'] == 0


def test_paired_report_clusters_seats_and_does_not_invent_one_deal_interval(policy, tmp_path, local_inventory):
    from brass_ai.replay_archive import Archive
    m = controller(); suite = m.prepare_suite(policy, SearchConfig(), master_seed=8)
    rows = []
    for spec in suite['games']:
        rank = [1]*4; vp = [5]*4
        if spec['treatment'] == 'search':
            rank[spec['candidate_seat']] = 0; vp[spec['candidate_seat']] = 10
        else:
            rank[(spec['candidate_seat']+1)%4] = 0
        rows.append(dict(m._lineage(suite, spec), ranking=rank, vp=vp, plies=0, telemetry=None, recovered=True))
    report = m.build_report(tmp_path/'report', suite, rows, Archive(), 'complete')
    assert report['independent_complete_deals'] == 1
    assert report['paired_difference']['win'] == {'mean': 1., 'ci95': None}
    assert report['paired_difference']['vp']['mean'] == 5.
    assert report['costs']['missing_telemetry_games'] == 8
    assert report['summaries']['search']['independent_deals'] == 1


def test_seed_collision_from_unified_archive_and_training_journal(policy, local_inventory, monkeypatch, tmp_path):
    import gzip
    m = controller()
    initial = m.prepare_suite(policy, SearchConfig(), master_seed=4321)
    archived = initial['deal_seeds'][0]
    root = tmp_path/'archive'; (root/'games').mkdir(parents=True)
    (root/'games'/'old.json.gz').write_bytes(gzip.compress(json.dumps({'initial': {'seed': archived}}).encode()))
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(root))
    second = m.prepare_suite(policy, SearchConfig(), master_seed=4321)
    assert archived not in second['deal_seeds']
    (local_inventory/'journal.jsonl').write_text(json.dumps({'deal_seed': second['deal_seeds'][0]})+'\n'+ '{"truncated":')
    third = m.prepare_suite(policy, SearchConfig(), master_seed=4321)
    assert not set(third['deal_seeds']) & {archived, second['deal_seeds'][0]}
    assert third['seed_inventory']['count'] == 2


def test_stop_during_act_does_not_commit_pending_move_or_count_broadcast(policy, tmp_path, local_inventory, monkeypatch):
    from brass_ai.replay_archive import Archive
    m = controller(); out = tmp_path/'run'; stopped = [False]
    original_plain, original_search = m.TorchAgent.act, m.SearchAgent.act
    def plain(self, obs):
        result = original_plain(self, obs); stopped[0] = True; return result
    def search(self, obs):
        result = original_search(self, obs); stopped[0] = True; return result
    monkeypatch.setattr(m.TorchAgent, 'act', plain)
    monkeypatch.setattr(m.SearchAgent, 'act', search)
    report = m.evaluate_search(policy, SearchConfig(max_rollouts=0), out, master_seed=55,
                               stop_requested=lambda: stopped[0])
    assert report['stop_reason'] == 'requested'
    assert len(Archive().records()) == 1 and Archive().records()[0]['moves'] == []
    assert report['costs']['partial_decisions'] == 1
    assert report['costs']['unapplied_decisions'] == 1
    assert report['costs']['real_memory_update_calls'] == 0
