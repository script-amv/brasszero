# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from brass_rl import four_branch_study as study


def test_interrupted_controller_resumes_latest_and_waits_for_all_endpoints(tmp_path, monkeypatch):
    from brass_rl import train as trainer
    from brass_rl.config import TrainConfig
    from dataclasses import asdict
    import json
    from brass_rl.experiments import writer_lock
    monkeypatch.setattr(study, 'BASE', 10)
    monkeypatch.setattr(study, 'END', 12)
    monkeypatch.setattr(study, 'storage', lambda *args, **kwargs: {})
    monkeypatch.setattr(study, 'caffeinate', lambda pid: None)
    for arm in study.ARMS:
        (tmp_path/arm/'updates/000010').mkdir(parents=True)
        (tmp_path/arm/'config.json').write_text(json.dumps(asdict(TrainConfig())))
    units = []
    def advance(cfg, dest, resume, max_updates, stop_requested):
        assert resume == dest/'latest'
        start = study.latest(tmp_path, dest.name)
        with writer_lock(dest):
            for number in range(start+1, max_updates+1):
                stop_requested()
                (dest/'updates'/('%06d' % number)).mkdir()
                units.append((dest.name, number))
        return SimpleNamespace(completed_updates=max_updates)
    monkeypatch.setattr(trainer, 'train', advance)
    def gate(root, manifest):
        assert all(study.latest(root, a) == 12 for a in study.ARMS)
        return {'parent': {}}
    calls = []
    monkeypatch.setattr('brass_rl.four_branch_report.markdown_report', lambda result: 'fixture')
    monkeypatch.setattr(study, 'verify_training', gate)
    def final(root, endpoints, manifest, check):
        calls.append('eval')
        return {}
    monkeypatch.setattr(study, 'final_evaluation', final)
    status = study.run(tmp_path, {}, lambda: len(units) == 3)
    assert status['status'] == 'stopped' and calls == []
    assert status['completed_updates'] == {'control': 12, 'A': 11, 'B': 10, 'C': 10}
    result = study.run(tmp_path, {}, lambda: False)
    assert result['status'] == 'complete'
    assert units == [(a, u) for a in study.ARMS for u in (11, 12)]
    assert study.run(tmp_path, {}, lambda: False) == result and calls == ['eval']


def test_disk_guard_reserves_space_for_a_whole_unit(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(study, 'disk_bytes', lambda *roots: study.LIMIT)
    with pytest.raises(study.StudyStopped, match='disk'):
        study.storage(tmp_path, {'shared_baseline_bytes':study.LIMIT}, check=True)


def test_prior_training_seeds_are_also_excluded(tmp_path, monkeypatch):
    import pytest
    path = tmp_path/'training/runs/earlier/updates/000001'
    path.mkdir(parents=True)
    import json
    (path/'manifests.json').write_text(json.dumps({'train': [{'deal_seed': 2**64-1}]}))
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='historical training'):
        study.historical_training_evidence(tmp_path/'study', [2**64-1], 41)


def test_published_policy_bytes_verified_at_each_boundary(tmp_path):
    import pytest
    from brass_rl.four_branch_study import verify_public_copy
    first, gui = tmp_path/'local', tmp_path/'public'
    for path in (first, gui):
        path.mkdir()
        for name in ('policy.pt', 'manifest.json', 'metadata.json'):
            (path/name).write_bytes(name.encode())
    verify_public_copy(first, gui)
    (gui/'policy.pt').write_bytes(b'broken')
    with pytest.raises(ValueError, match='published policy'):
        verify_public_copy(first, gui)


def test_parent_evaluation_is_owned_by_study_and_preserves_source_experiment():
    from brass_rl.four_branch_study import evaluation_origin
    parent = {'experiment_id': 'historic-parent'}
    branch = {'experiment_id': 'distinct-child'}
    p = evaluation_origin('parent', parent)
    assert p['source_experiment'] == 'historic-parent'
    assert p['run_experiment'] != 'historic-parent'
    assert p['study_arm'] == 'parent'
    b = evaluation_origin('B', branch)
    assert b['run_experiment'] == b['source_experiment'] == 'distinct-child'


def test_training_recorder_uses_seat_identities_for_collection_policy():
    from brass_rl.four_branch_study import check_training_replay
    import pytest
    labels = ['collection-policy', 'training-000010:weights-sha256:abc', 'greedy', 'random']
    replay = dict(initial={'seed': 2**64-1}, result={'vp':[4, 3, 2, 1], 'ranking':[0, 1, 2, 3]},
                  provenance={'experiment':'branch', 'update':2, 'game':2},
                  seats=[{'name': label} for label in labels], moves=[])
    game = dict(game_id=2, deal_seed=2**64-1, outcome=replay['result'], plies=0,
                lineup=[{'kind':'current'}, {'kind':'frozen', 'snapshot_id':'s'}, {'kind':'greedy'}, {'kind':'random'}])
    history = {'s': {'role':'training', 'update':10, 'sha256':'abc'}}
    check_training_replay(replay, game, 2, 2**64-1, 'branch', 2, 'collection-policy', history)
    replay['seats'][0]['name'] = 'different-policy'
    with pytest.raises(ValueError, match='seat identity'):
        check_training_replay(replay, game, 2, 2**64-1, 'branch', 2, 'collection-policy', history)


def test_matching_parent_copies_cannot_exchange_rotations_and_outcomes():
    import pytest
    from brass_rl.four_branch_study import check_evaluation_replay
    origin = dict(run_experiment='comparison', source_experiment='source', study_arm='parent')
    identity = 'identical-frozen-parent'
    common = dict(lineup='frozen688', deal_id=0, update=688, deal_seed=2**64-1,
                  labels=[identity]*4, candidate_id=identity, suite_id='fixed', plies=2)
    rows = [dict(common, game_id=i, rotation=i, candidate_seat=(-i)%4,
                 vp=vp, ranking=ranks, replay_id='record-'+str(i))
            for i, (vp, ranks) in enumerate([([4, 3, 2, 1], [0, 1, 2, 3]),
                                             ([1, 2, 3, 4], [3, 2, 1, 0])])]
    def recorded(row):
        return dict(game_id=row['replay_id'], initial={'seed':row['deal_seed']}, seats=[{'name':identity}]*4,
                    moves=[{}, {}], result={'vp':row['vp'], 'ranking':row['ranking']},
                    provenance=dict(origin, experiment='fixed', candidate_id=identity, update=688,
                                    game=row['game_id'], rotation=row['rotation'], deal=0, lineup='frozen688'))
    for row in rows:
        check_evaluation_replay(recorded(row), row, origin)
    swapped = dict(rows[0], **{k: rows[1][k] for k in ('replay_id', 'vp', 'ranking')})
    with pytest.raises(ValueError, match='provenance'):
        check_evaluation_replay(recorded(rows[1]), swapped, origin)


def test_training_audit_rejects_exchanged_replay_filenames(tmp_path):
    import gzip
    import json
    import pytest
    from brass_ai.replay_archive import Archive
    from brass_rl.four_branch_study import training_replays_by_game
    archive = Archive(tmp_path)
    ids = ('a'*64, 'b'*64)
    records = [{'game_id': rid, 'provenance': {'game': number}} for number, rid in enumerate(ids)]
    payloads = [gzip.compress(json.dumps(record).encode()) for record in records]
    for rid, data in zip(ids, payloads):
        archive.path(rid).write_bytes(data)
    assert training_replays_by_game(archive, ids) == dict(enumerate(records))
    for rid, data in zip(ids, reversed(payloads)):
        archive.path(rid).write_bytes(data)
    with pytest.raises(ValueError, match='training replay ID'):
        training_replays_by_game(archive, ids)
