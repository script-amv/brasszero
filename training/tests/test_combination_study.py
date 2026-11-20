# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from pathlib import Path

import pytest

from brass_rl.config import TrainConfig


def test_protocol_pairs_same_training_deals_and_separates_source_from_opponents():
    from brass_rl import combination_study as study
    cfg = TrainConfig()
    control, bc, abc = (study.arm_config(cfg, name) for name in study.ARMS)
    assert control == cfg
    assert bc == replace(control, ppo=replace(cfg.ppo, reward='score', critic_target='terminal'))
    assert abc == replace(bc, league=replace(bc.league, lineup_probabilities=(.8, .2, 0), lineup_schedule='four_of_five'))
    assert (study.BASE, study.END, study.FIRST_GAME, study.GAMES) == (738, 788, 1476, 100)
    assert (study.END-study.BASE)*cfg.games_per_update*len(study.ARMS) == 300
    suite = study.suite(study.SOURCE_ID)
    frozen = suite.lineups['frozen688']
    assert frozen[0] == 'candidate' and all(x == frozen[1] for x in frozen[1:])
    assert Path(frozen[1]['path']).parts[-3:] == ('kaggle-12h-2026-11-18','updates','000688')
    assert frozen[1]['id'].startswith('policy-sha256-v1:e0059c09') and frozen[1]['id'] != suite.candidate_id
    assert len(suite.lineups)*suite.deals*4*len(study.CANDIDATES) == 256
    assert len(set(study.head_protocol()['deal_seeds']+suite.manifest()['deal_seeds'])) == 10


def test_head_lineups_pin_true_updates_and_ids_at_all_seats():
    from brass_rl import combination_study as study
    endpoints = {name: {'id': name+'-exact', 'update': 738 if name == 'parent' else 788}
                 for name in study.CANDIDATES}
    games = study.head_schedule(study.head_protocol(), endpoints)
    assert len(games) == len(set(g['replay_id'] for g in games)) == 8
    for g in games:
        assert [endpoints[name]['id'] for name in g['arms']] == g['seat_ids']
        assert g['seat_updates'] == [738 if name == 'parent' else 788 for name in g['arms']]
    for seat in range(4):
        assert sorted(g['arms'][seat] for g in games) == sorted(list(endpoints)*2)


def test_head_to_head_resumes_archive_after_stop_and_missing_last_scalar(tmp_path, monkeypatch):
    """A published replay is the durable boundary even before its result row is written."""
    import brass
    from brass_ai.agents import register
    from brass_ai.replay_archive import Archive
    from brass_rl import combination_study as study
    from brass_rl import agent as player
    from brass_rl.learning_study import StudyStopped
    protocol = study.head_protocol()
    # Keep fixture deals separate from the prospective scientific deals.
    protocol.update(deal_seeds=[160, 161], suite='test-two-blocks')
    endpoints = {name: dict(path=str(tmp_path/name), id='fixture-'+name, update=n)
                 for name, n in zip(study.CANDIDATES, protocol['updates'])}
    class StubPolicy:
        def __init__(self, seed, identity):
            self.bot = brass.RandomBot(seed)
            self.replay_identity = identity

        def act(self, observation):
            return self.bot.act(observation)

    # Model loading is the only substituted operation; native games and archive commits are real.
    def register_stub(name, path, **kwargs):
        register(name, lambda seed: StubPolicy(seed, endpoints[name]['id']))
    monkeypatch.setattr(player, 'register_checkpoint', register_stub)
    calls = []
    def interrupt(arm):
        if len(calls) == 3:
            raise StudyStopped('test boundary')
        calls.append(arm)
    with pytest.raises(StudyStopped):
        study.head_to_head(tmp_path, endpoints, protocol, interrupt)
    assert len(list((tmp_path/'head-to-head/games').glob('*.json'))) == 3
    (tmp_path/'head-to-head/games/02.json').unlink()
    result = study.head_to_head(tmp_path, endpoints, protocol, calls.append)
    assert len(calls) == 8 and len(result['games']) == 8
    assert study.head_to_head(tmp_path, endpoints, protocol, lambda _: pytest.fail('duplicate')) == result
    archive = Archive()
    assert len(list(archive.games.glob('*.json.gz'))) == 8
    for game in result['games']:
        replay = archive.read(game['replay_id'])
        assert replay['provenance']['seat_updates'] == game['seat_updates']
        assert 'update' not in replay['provenance']
        assert [s['bot'] for s in replay['seats']] == game['seat_ids']


def test_three_branch_controller_resumes_latest_before_any_frozen_game(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    from brass_rl import combination_study as study
    from brass_rl import train as trainer
    from brass_rl import four_branch_study as shared
    for arm in study.ARMS:
        (tmp_path/arm/'updates/000738').mkdir(parents=True)
        (tmp_path/arm/'config.json').write_text(json.dumps(asdict(TrainConfig())))
    units, frozen = [], []
    def advance(cfg, dest, resume, max_updates, stop_requested):
        assert resume == dest/'latest'
        for n in range(shared.latest(tmp_path, dest.name)+1, max_updates+1):
            stop_requested()
            (dest/'updates'/('%06d'%n)).mkdir()
            units.append((dest.name, n))
        return SimpleNamespace(completed_updates=max_updates)
    def verify(root, manifest):
        assert all(shared.latest(root, a) == 788 for a in study.ARMS)
        return {}
    def evaluate(root, endpoints, manifest, check):
        frozen.append(1)
        return {}
    monkeypatch.setattr(trainer, 'train', advance)
    monkeypatch.setattr(shared, 'caffeinate', lambda _: None)
    monkeypatch.setattr(shared, 'storage', lambda *a, **k: {})
    monkeypatch.setattr(study, 'verify_training', verify)
    monkeypatch.setattr(study, 'final_evaluation', evaluate)
    monkeypatch.setattr(study, 'render', lambda _: '')
    stopped = study.run(tmp_path, {}, lambda: len(units) == 52)
    assert stopped['status'] == 'stopped' and frozen == []
    assert stopped['completed_updates'] == {'control': 788, 'BC': 740, 'ABC': 738}
    complete = study.run(tmp_path, {}, lambda: False)
    assert complete['new_training_games'] == 300 and complete['status'] == 'complete'
    assert units == [(name, n) for name in study.ARMS for n in range(739, 789)]
    assert study.run(tmp_path, {}, lambda: False) == complete and len(frozen) == 1


def test_sealed_prior_deals_are_excluded_even_before_a_replay_exists(tmp_path, monkeypatch):
    from brass_rl import four_branch_study as shared
    from brass_rl.learning_study import fixed
    old = tmp_path/'training/runs/prior/protocol.json'
    fixed(old, {'deal_seeds': [2**64-1]})
    archive = tmp_path/'archive'
    monkeypatch.setattr(shared, 'ROOT', tmp_path)
    monkeypatch.setattr(shared, 'shared_roots', lambda: (archive,tmp_path/'models'))
    with pytest.raises(ValueError, match='previously used'):
        shared.freshness_evidence(tmp_path/'new', [2**64-1]+list(range(9)), list(range(100,200)),
                                  expected_deals=10, extra_protocols=(old,))


@pytest.mark.parametrize('seed', [991, 992])
def test_embedded_ancestor_training_and_eval_deals_are_excluded(tmp_path, monkeypatch, seed):
    from brass_rl import four_branch_study as shared
    from brass_rl.learning_study import fixed
    old = tmp_path/'training/runs/old/ancestry/parent-journal/000002/manifests.json'
    fixed(old, {'train': [{'deal_seed': 991}], 'dev': {'deal_seeds': [992]}})
    monkeypatch.setattr(shared, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='historical'):
        shared.historical_training_evidence(tmp_path/'new', [seed], 21, include_ancestry=True)


def test_matched_recovery_verifies_ordered_seats_and_never_resamples(tmp_path, monkeypatch):
    import json
    import torch
    from brass_ai.replay_archive import Archive, native_id
    from brass_rl import combination_evaluation as matched
    from brass_rl.artifacts import export_model_policy
    from brass_rl.config import ModelConfig
    from brass_rl.evaluate import EvalSuite
    from brass_rl.model import BrassPolicy
    from brass_rl.policy_io import inspect_policy
    from brass_rl.schema import ComponentData

    torch.set_num_threads(1)
    components = ComponentData.load()
    model = BrassPolicy(ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4, message_passing_rounds=1),
                        components.schema, components)
    paths = {n: export_model_policy(model, n, tmp_path/str(n)) for n in (688, 738, 788)}
    ids = {n: inspect_policy(path).id for n, path in paths.items()}
    frozen = {'id': ids[688], 'path': str(paths[688])}
    lineups = {'greedy': ['candidate']+['greedy']*3, 'frozen688': ['candidate']+[frozen]*3}
    archive = Archive()
    for n in (738, 788):
        declared = EvalSuite('dev', 916355, 1, lineups, ids[n], replay_limit=0, outcomes_only=True)
        out = tmp_path/('eval-%d' % n)
        rows = matched.evaluate(paths[n], declared, out).results
        for row in rows:
            seat = (-row['rotation']) % 4
            frozen_lineup = row['lineup'] == 'frozen688'
            updates = [688 if frozen_lineup else None]*4
            labels = [ids[688] if frozen_lineup else native_id('greedy')]*4
            updates[seat], labels[seat] = n, ids[n]
            record = archive.read(row['replay_id'])
            assert row['seat_updates'] == record['provenance']['seat_updates'] == updates
            assert row['seat_ids'] == record['provenance']['seat_ids'] == labels
            assert [s['bot'] for s in record['seats']] == labels
        last = rows[-1]
        row_path = out/'candidates'/ids[n]/'games'/'00000007.json'
        before = archive.path(last['replay_id']).read_bytes()
        row_path.unlink()
        with monkeypatch.context() as patch:
            patch.setattr(matched, 'play_game', lambda *a, **kw: pytest.fail('resampled an archived outcome'))
            assert matched.evaluate(paths[n], declared, out).results[-1] == last
            # The native archive can also commit before the first candidate report or row.
            row_path.unlink()
            (out/'candidates'/ids[n]/'report.json').unlink()
            rebuilt = matched.evaluate(paths[n], declared, out).results[-1]
            assert rebuilt == dict(last, seconds=None)
            assert archive.path(last['replay_id']).read_bytes() == before
            original_read = Archive.read
            def wrong_provenance(archive, replay_id):
                record = original_read(archive, replay_id)
                if replay_id == rebuilt['replay_id']:
                    record['provenance']['seat_updates'] = [688]*4
                return record
            with patch.context() as corrupt:
                corrupt.setattr(Archive, 'read', wrong_provenance)
                with pytest.raises(ValueError, match='provenance'):
                    matched.evaluate(paths[n], declared, out)
            wrong = dict(rebuilt, seat_updates=[688]*4)
            row_path.write_text(json.dumps(wrong))
            with pytest.raises(ValueError):
                matched.evaluate(paths[n], declared, out)
    assert len(list(archive.games.glob('*.json.gz'))) == 16


@pytest.mark.parametrize('stop_mode', ['disk', 'manual'])
def test_initialization_is_timed_and_can_resume_a_stop_after_one_fork(tmp_path, monkeypatch, stop_mode):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    from brass_rl import combination_study as study
    from brass_rl import four_branch_study as shared
    from brass_rl import artifacts, train as trainer
    from brass_rl.learning_study import StudyStopped, read, fixed
    ticks = [500.]
    power = []
    class Power:
        def terminate(self):
            power.append('released')
        def wait(self):
            pass
    monkeypatch.setattr(shared, 'time', SimpleNamespace(time=lambda: ticks[0], monotonic=lambda: ticks[0]))
    monkeypatch.setattr(shared, 'caffeinate', lambda _: power.append('held') or Power())
    cfg = TrainConfig()
    parent = {'experiment_id': 'original', 'model_id': 'frozen738'}
    definition = {'arms': list(study.ARMS), 'parent': parent, 'base_update': 738, 'first_game_id': 1476,
                  'shared_baseline_bytes': 0}
    forks = []
    def fork(source, dest, lam, study_config):
        ticks[0] += 10
        forks.append(dest.name)
        (dest/'updates/000738').mkdir(parents=True)
        (dest/'config.json').write_text(json.dumps(asdict(study_config)))
    monkeypatch.setattr(shared, 'fork_arm', fork)
    monkeypatch.setattr(shared, 'read_experiment', lambda dest: dict(parent=parent, initial_state_verified={'all': True},
                         id=dest.name, initial_model_id='frozen738'))
    monkeypatch.setattr(artifacts, 'load_update', lambda p, config: SimpleNamespace(state=SimpleNamespace(
        next_game_id=1476, manifests={'experiment_id': p.parents[1].name})))
    monkeypatch.setattr(trainer, 'publish_policies', lambda dest: None)
    low_disk = [True]
    def storage(root, manifest, *, check=False):
        if check and low_disk[0] and forks == ['control']:
            raise StudyStopped('disk reserve reached')
        return {}
    monkeypatch.setattr(shared, 'storage', storage)
    def initialize(root, *, checkpoint):
        assert read(root/'progress.json')['phase'] == 'initialization' and power[-1] == 'held'
        ticks[0] += 5
        checkpoint(None)
        fixed(root/'study.json', definition)
        checkpoint(None, definition)
        shared.prepare_branches(root, definition, tmp_path/'source', cfg, study.arm_config, checkpoint=checkpoint)
        return definition
    monkeypatch.setattr(study, 'initialize', initialize)
    first = study.run(tmp_path, None, lambda: stop_mode == 'manual' and bool(forks), init_only=True)
    assert first['status'] == 'stopped' and first['phase'] == 'initialization'
    assert stop_mode in first['reason']
    assert first['active_seconds'] == 15 and first['new_training_games'] == 0
    assert forks == ['control'] and power == ['held','released']
    ticks[0] += 20
    low_disk[0] = False
    resumed = study.run(tmp_path, None, lambda: False, init_only=True)
    assert forks == ['control','BC','ABC'] and resumed['status'] == 'initialized'
    assert resumed['active_seconds'] == 40 and resumed['elapsed_seconds'] == 60
    assert resumed['invocation_count'] == 2 and power == ['held','released','held','released']


def test_failed_initialization_is_visible_and_abrupt_work_is_marked_unknown(tmp_path, monkeypatch):
    from brass_rl import combination_study as study
    from brass_rl import four_branch_study as common
    from brass_rl.learning_study import read, write
    monkeypatch.setattr(common, 'caffeinate', lambda _: None)
    def fail(root, *, checkpoint):
        assert read(root/'progress.json')['phase'] == 'initialization'
        raise ValueError('unreadable declaration')
    monkeypatch.setattr(study, 'initialize', fail)
    write(tmp_path/'progress.json', dict(status='running', phase='training', arm='BC',
                                       active_seconds=13, updated_epoch=200))
    with pytest.raises(ValueError, match='unreadable'):
        study.run(tmp_path, None, lambda: False, init_only=True)
    status = read(tmp_path/'progress.json')
    assert status['status'] == 'failed' and status['phase'] == 'initialization'
    gap, = status['interrupted_intervals']
    assert gap['last_phase'] == 'training' and gap['last_arm'] == 'BC'
    assert gap['last_observed_epoch'] == 200 and gap['next_started_epoch'] == status['invocation_started_epoch']
    assert status['active_seconds'] >= 13 and 'unreadable' in status['reason']


def test_init_cli_reports_status_before_entering_initializer(tmp_path, monkeypatch, capsys):
    import json
    from brass_rl import combination_study as study
    from brass_rl import four_branch_study as common
    from brass_rl.learning_study import read
    monkeypatch.setattr(study, 'STUDY', tmp_path)
    monkeypatch.setattr(common, 'caffeinate', lambda _: None)
    monkeypatch.setattr(common, 'storage', lambda *args, **kwargs: {})
    def initialize(root, *, checkpoint):
        assert read(root/'progress.json')['phase'] == 'initialization'
        checkpoint(None)
        return {'shared_baseline_bytes': 0}
    monkeypatch.setattr(study, 'initialize', initialize)
    study.main(['init'])
    assert json.loads(capsys.readouterr().out)['status'] == 'initialized'
    assert read(tmp_path/'progress.json')['new_training_games'] == 0
